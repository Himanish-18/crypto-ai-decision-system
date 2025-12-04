import { motion } from 'framer-motion'
import { twMerge } from 'tailwind-merge'

interface NeonButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: 'primary' | 'danger' | 'success'
    isLoading?: boolean
}

export function NeonButton({
    children,
    className,
    variant = 'primary',
    isLoading,
    ...props
}: NeonButtonProps) {
    const baseStyles = "relative px-6 py-3 rounded-lg font-bold uppercase tracking-wider transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"

    const variants = {
        primary: "bg-indigo-600 text-white hover:bg-indigo-500 shadow-[0_0_15px_rgba(79,70,229,0.5)] hover:shadow-[0_0_25px_rgba(79,70,229,0.8)] border border-indigo-400/30",
        danger: "bg-red-600 text-white hover:bg-red-500 shadow-[0_0_15px_rgba(220,38,38,0.5)] hover:shadow-[0_0_25px_rgba(220,38,38,0.8)] border border-red-400/30",
        success: "bg-green-600 text-white hover:bg-green-500 shadow-[0_0_15px_rgba(22,163,74,0.5)] hover:shadow-[0_0_25px_rgba(22,163,74,0.8)] border border-green-400/30",
    }

    return (
        <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className={twMerge(baseStyles, variants[variant], className)}
            disabled={isLoading}
            {...props}
        >
            {isLoading ? (
                <div className="flex items-center justify-center">
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                </div>
            ) : children}
        </motion.button>
    )
}
