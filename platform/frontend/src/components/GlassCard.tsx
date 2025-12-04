import { motion } from 'framer-motion'
import { clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

interface GlassCardProps {
    children: React.ReactNode
    className?: string
    hoverEffect?: boolean
    floating?: boolean
}

export function GlassCard({ children, className, hoverEffect = false, floating = false }: GlassCardProps) {
    const floatingAnimation = floating ? {
        y: [0, -10, 0],
        transition: {
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut"
        }
    } : {}

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0, ...floatingAnimation }}
            whileHover={hoverEffect ? { scale: 1.02, boxShadow: "0 0 30px rgba(79, 70, 229, 0.3)" } : undefined}
            className={twMerge(
                "bg-gray-900/40 backdrop-blur-md border border-white/10 rounded-xl p-6 shadow-xl relative z-10",
                className
            )}
        >
            {children}
        </motion.div>
    )
}
