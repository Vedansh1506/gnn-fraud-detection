/**
 * The flagged account's money-flow neighbourhood (Design Doc §4.3).
 *
 * A force layout drawn as SVG. d3-force computes positions; React owns the DOM.
 * Mixing the two - letting d3 select and mutate nodes React also renders - is
 * the classic way these integrations break, so d3 here is used purely as a
 * physics solver.
 *
 * This is a context view, not a graph explorer: the API caps the neighbourhood
 * and says so, and the panel repeats that cap rather than presenting a
 * truncated picture as the whole story. The SHAP explanation beside it carries
 * the "why" in words, so a failed or unread graph never hides anything
 * essential (§11).
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from 'd3-force'
import { motion } from 'motion/react'
import type { GraphResponse } from '../lib/api'
import { formatAmount, shortAccount } from '../lib/format'

interface FlowNode extends SimulationNodeDatum {
  id: string
  isFocus: boolean
  /** Touched by at least one edge the dataset labels as laundering. */
  isFlagged: boolean
  degree: number
}

type FlowLink = SimulationLinkDatum<FlowNode> & {
  txId: string
  amount: number
  isLaundering: boolean
}

const WIDTH = 560
const HEIGHT = 300

/**
 * Force settings scaled to the neighbourhood size.
 *
 * A fixed link distance is wrong at both ends: with six accounts the layout
 * huddles in the middle of a mostly empty box, and with thirty it overflows
 * the frame. Spreading small graphs out and pulling large ones in keeps the
 * drawing filling roughly the same area either way.
 */
function layoutForces(nodeCount: number) {
  if (nodeCount <= 8) return { distance: 108, charge: -640, collide: 30 }
  if (nodeCount <= 16) return { distance: 88, charge: -420, collide: 27 }
  return { distance: 66, charge: -260, collide: 22 }
}

export function MoneyFlowGraph({
  graph,
  focusAccount,
}: {
  graph: GraphResponse
  focusAccount: string
}) {
  const [hovered, setHovered] = useState<string | null>(null)
  const frameRef = useRef<number>(0)

  const { nodes, links } = useMemo(() => {
    const flaggedAccounts = new Set<string>()
    for (const edge of graph.edges) {
      if (edge.is_laundering) {
        flaggedAccounts.add(edge.source)
        flaggedAccounts.add(edge.target)
      }
    }

    const degrees = new Map<string, number>()
    for (const edge of graph.edges) {
      degrees.set(edge.source, (degrees.get(edge.source) ?? 0) + 1)
      degrees.set(edge.target, (degrees.get(edge.target) ?? 0) + 1)
    }

    const nodeList: FlowNode[] = graph.nodes.map((id) => ({
      id,
      isFocus: id === focusAccount,
      isFlagged: flaggedAccounts.has(id),
      degree: degrees.get(id) ?? 0,
    }))

    const linkList: FlowLink[] = graph.edges.map((edge) => ({
      source: edge.source,
      target: edge.target,
      txId: edge.tx_id,
      amount: edge.amount_paid,
      isLaundering: edge.is_laundering,
    }))

    return { nodes: nodeList, links: linkList }
  }, [graph, focusAccount])

  // Positions live in state so React re-renders as the simulation settles.
  const [positions, setPositions] = useState<FlowNode[]>([])

  useEffect(() => {
    if (nodes.length === 0) return

    // Copies: the simulation mutates what it is given, and mutating the memo'd
    // arrays would make the render output depend on simulation ticks.
    const simNodes = nodes.map((node) => ({ ...node }))
    const simLinks = links.map((link) => ({ ...link }))

    const forces = layoutForces(simNodes.length)
    const simulation = forceSimulation<FlowNode>(simNodes)
      .force(
        'link',
        forceLink<FlowNode, FlowLink>(simLinks)
          .id((node) => node.id)
          .distance(forces.distance)
          .strength(0.3),
      )
      .force('charge', forceManyBody().strength(forces.charge))
      .force('center', forceCenter(WIDTH / 2, HEIGHT / 2))
      .force('collide', forceCollide(forces.collide))
      .alphaDecay(0.05)

    // Pin the focus account centrally: the whole point of the view is "this
    // account and what surrounds it", which is lost if it drifts to an edge.
    const focus = simNodes.find((node) => node.isFocus)
    if (focus) {
      focus.fx = WIDTH / 2
      focus.fy = HEIGHT / 2
    }

    simulation.on('tick', () => {
      // Throttle to animation frames - the simulation ticks faster than the
      // screen refreshes and setState per tick is wasted work.
      cancelAnimationFrame(frameRef.current)
      frameRef.current = requestAnimationFrame(() => setPositions([...simNodes]))
    })

    return () => {
      simulation.stop()
      cancelAnimationFrame(frameRef.current)
    }
  }, [nodes, links])

  const positioned = positions.length > 0 ? positions : nodes
  const byId = new Map(positioned.map((node) => [node.id, node]))

  return (
    <div>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        // Capped height so a small neighbourhood doesn't leave a tall empty
        // panel; centred rather than stretched.
        className="mx-auto block max-h-[300px] w-full"
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={`Money-flow neighbourhood for ${focusAccount}: ${graph.nodes.length} accounts and ${graph.edges.length} payments.`}
      >
        <defs>
          {/* markerUnits="userSpaceOnUse" keeps the head a fixed size instead
              of scaling with each line's stroke width, which made the
              laundering edges' arrows noticeably chunkier than the rest. */}
          <marker
            id="flow-arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            markerUnits="userSpaceOnUse"
            orient="auto-start-reverse"
          >
            <path d="M0,1.5 L9,5 L0,8.5 Z" fill="var(--ink-muted)" opacity="0.55" />
          </marker>
        </defs>

        {links.map((link) => {
          const source = byId.get(
            typeof link.source === 'string' ? link.source : (link.source as FlowNode).id,
          )
          const target = byId.get(
            typeof link.target === 'string' ? link.target : (link.target as FlowNode).id,
          )
          if (!source || !target) return null

          const isActive =
            hovered !== null && (hovered === source.id || hovered === target.id)

          return (
            <line
              key={link.txId}
              x1={source.x ?? 0}
              y1={source.y ?? 0}
              x2={target.x ?? 0}
              y2={target.y ?? 0}
              stroke={link.isLaundering ? 'var(--risk-critical)' : 'var(--ink-muted)'}
              strokeWidth={link.isLaundering ? 1.8 : 1}
              // Known-laundering edges are the signal; everything else is
              // context and recedes so it doesn't compete with them.
              strokeOpacity={isActive ? 0.95 : link.isLaundering ? 0.7 : 0.28}
              markerEnd="url(#flow-arrow)"
            >
              <title>
                {`${link.txId}: ${formatAmount(link.amount, null)}${
                  link.isLaundering ? ' — labelled laundering' : ''
                }`}
              </title>
            </line>
          )
        })}

        {positioned.map((node, index) => {
          const radius = node.isFocus ? 11 : 6 + Math.min(node.degree, 6) * 0.7
          const color = node.isFocus
            ? 'var(--accent)'
            : node.isFlagged
              ? 'var(--risk-critical)'
              : 'var(--ink-muted)'

          return (
            <motion.g
              key={node.id}
              initial={{ opacity: 0, scale: 0.6 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: Math.min(index, 20) * 0.015, duration: 0.3 }}
              onMouseEnter={() => setHovered(node.id)}
              onMouseLeave={() => setHovered(null)}
              style={{ cursor: 'default' }}
            >
              {node.isFocus && (
                <circle
                  cx={node.x ?? 0}
                  cy={node.y ?? 0}
                  r={radius + 7}
                  fill="none"
                  stroke="var(--accent)"
                  strokeOpacity={0.35}
                  strokeWidth={1}
                />
              )}
              <circle
                cx={node.x ?? 0}
                cy={node.y ?? 0}
                r={radius}
                fill={color}
                fillOpacity={node.isFocus || node.isFlagged ? 0.95 : 0.55}
                // A 2px surface ring keeps overlapping nodes readable.
                stroke="var(--surface)"
                strokeWidth={2}
              />
              <title>{node.id}</title>

              {(node.isFocus || node.isFlagged || hovered === node.id) && (
                <text
                  x={node.x ?? 0}
                  y={(node.y ?? 0) + radius + 12}
                  textAnchor="middle"
                  className="identifier"
                  fontSize={9}
                  fill="var(--ink-secondary)"
                >
                  {shortAccount(node.id)}
                </text>
              )}
            </motion.g>
          )
        })}
      </svg>

      <GraphLegend graph={graph} />
    </div>
  )
}

function GraphLegend({ graph }: { graph: GraphResponse }) {
  const launderingEdges = graph.edges.filter((edge) => edge.is_laundering).length

  return (
    <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-[var(--border)] pt-3 text-[11px] text-[var(--ink-secondary)]">
      <LegendKey color="var(--accent)" label="This account" />
      <LegendKey color="var(--risk-critical)" label="Touches labelled laundering" />
      <LegendKey color="var(--ink-muted)" label="Other counterparty" />
      <span className="ml-auto text-[var(--ink-muted)]">
        {graph.nodes.length} accounts · {graph.edges.length} payments
        {launderingEdges > 0 && ` · ${launderingEdges} labelled`}
        {graph.truncated && ' · view capped'}
      </span>
    </div>
  )
}

function LegendKey({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span
        className="inline-block h-2.5 w-2.5 rounded-full"
        style={{ backgroundColor: color }}
        aria-hidden
      />
      {label}
    </span>
  )
}
