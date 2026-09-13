/**
 * The ambient background on the login screen.
 *
 * It is a money-flow network: accounts as nodes, payments as edges, with
 * occasional pulses travelling along the edges and one node lighting up as
 * "flagged". That is the actual subject of this product, so the decoration
 * carries a little meaning rather than being generic gradient blobs.
 *
 * Rendered to canvas rather than SVG - a few hundred moving elements would cost
 * a DOM node each in SVG, and this is background, not content. It is purely
 * decorative: `aria-hidden`, and it stops entirely under reduced-motion.
 */

import { useEffect, useRef } from 'react'

interface Node {
  x: number
  y: number
  vx: number
  vy: number
  radius: number
  flagged: boolean
}

interface Edge {
  from: number
  to: number
  /** Position of the travelling pulse, 0..1, or null when idle. */
  pulse: number | null
  pulseSpeed: number
  flagged: boolean
}

const NODE_COUNT = 46
const LINK_DISTANCE = 190
const PULSE_CHANCE = 0.004

export function FlowField() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const context = canvas.getContext('2d')
    if (!context) return

    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    let width = 0
    let height = 0
    let nodes: Node[] = []
    let edges: Edge[] = []
    let frame = 0

    const readColor = (name: string) =>
      getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#5b8def'

    function seed() {
      const rect = canvas!.getBoundingClientRect()
      const ratio = Math.min(window.devicePixelRatio || 1, 2)
      width = rect.width
      height = rect.height
      canvas!.width = width * ratio
      canvas!.height = height * ratio
      context!.setTransform(ratio, 0, 0, ratio, 0, 0)

      nodes = Array.from({ length: NODE_COUNT }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.14,
        vy: (Math.random() - 0.5) * 0.14,
        radius: 1.4 + Math.random() * 2.2,
        // A realistic minority: most accounts in this dataset are ordinary.
        flagged: Math.random() < 0.09,
      }))

      edges = []
      for (let i = 0; i < nodes.length; i += 1) {
        for (let j = i + 1; j < nodes.length; j += 1) {
          const distance = Math.hypot(nodes[i].x - nodes[j].x, nodes[i].y - nodes[j].y)
          if (distance < LINK_DISTANCE && Math.random() < 0.28) {
            edges.push({
              from: i,
              to: j,
              pulse: null,
              pulseSpeed: 0.004 + Math.random() * 0.006,
              flagged: nodes[i].flagged || nodes[j].flagged,
            })
          }
        }
      }
    }

    function draw() {
      const accent = readColor('--accent')
      const critical = readColor('--risk-critical')

      context!.clearRect(0, 0, width, height)

      for (const edge of edges) {
        const a = nodes[edge.from]
        const b = nodes[edge.to]
        const distance = Math.hypot(a.x - b.x, a.y - b.y)
        if (distance > LINK_DISTANCE) continue

        // Nearer pairs draw stronger, so the network reads as depth rather
        // than a flat mesh.
        const strength = 1 - distance / LINK_DISTANCE
        context!.beginPath()
        context!.moveTo(a.x, a.y)
        context!.lineTo(b.x, b.y)
        context!.strokeStyle = edge.flagged ? critical : accent
        context!.globalAlpha = strength * (edge.flagged ? 0.32 : 0.2)
        context!.lineWidth = 1
        context!.stroke()

        if (edge.pulse !== null) {
          const x = a.x + (b.x - a.x) * edge.pulse
          const y = a.y + (b.y - a.y) * edge.pulse
          context!.beginPath()
          context!.arc(x, y, 1.8, 0, Math.PI * 2)
          context!.fillStyle = edge.flagged ? critical : accent
          context!.globalAlpha = 0.85 * (1 - Math.abs(edge.pulse - 0.5) * 1.2)
          context!.fill()
        }
      }

      for (const node of nodes) {
        context!.beginPath()
        context!.arc(node.x, node.y, node.radius, 0, Math.PI * 2)
        context!.fillStyle = node.flagged ? critical : accent
        context!.globalAlpha = node.flagged ? 0.9 : 0.6
        context!.fill()

        if (node.flagged) {
          // A slow halo, so the eye lands on the flagged accounts - the same
          // thing the queue asks the analyst to do.
          const phase = (Math.sin(frame * 0.02 + node.x) + 1) / 2
          context!.beginPath()
          context!.arc(node.x, node.y, node.radius + 3 + phase * 4, 0, Math.PI * 2)
          context!.strokeStyle = critical
          context!.globalAlpha = 0.18 * (1 - phase)
          context!.lineWidth = 1
          context!.stroke()
        }
      }

      context!.globalAlpha = 1
    }

    function step() {
      frame += 1
      for (const node of nodes) {
        node.x += node.vx
        node.y += node.vy
        if (node.x < 0 || node.x > width) node.vx *= -1
        if (node.y < 0 || node.y > height) node.vy *= -1
      }
      for (const edge of edges) {
        if (edge.pulse === null) {
          if (Math.random() < PULSE_CHANCE) edge.pulse = 0
        } else {
          edge.pulse += edge.pulseSpeed
          if (edge.pulse > 1) edge.pulse = null
        }
      }
      draw()
      animation = requestAnimationFrame(step)
    }

    let animation = 0
    seed()

    if (reduceMotion) {
      draw() // A still frame: the composition without the movement.
    } else {
      animation = requestAnimationFrame(step)
    }

    const onResize = () => {
      seed()
      draw()
    }
    window.addEventListener('resize', onResize)

    return () => {
      cancelAnimationFrame(animation)
      window.removeEventListener('resize', onResize)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      className="pointer-events-none absolute inset-0 h-full w-full"
    />
  )
}
