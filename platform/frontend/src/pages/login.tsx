import { useState } from 'react'
import { useRouter } from 'next/router'
import Head from 'next/head'
import dynamic from 'next/dynamic'
import { motion } from 'framer-motion'
import { NeonButton } from '@/components/NeonButton'
import { GlassCard } from '@/components/GlassCard'
import { auth } from '@/services/api'

const ThreeGlobe = dynamic(() => import('@/components/ThreeGlobe'), { ssr: false })

export default function Login() {
    const router = useRouter()
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault()
        setLoading(true)
        setError('')
        try {
            const { access_token } = await auth.login(email, password)
            localStorage.setItem('token', access_token)
            router.push('/dashboard')
        } catch (err) {
            setError('Invalid credentials')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen bg-black flex items-center justify-center relative overflow-hidden">
            <Head>
                <title>Login | Crypto AI</title>
            </Head>

            {/* Background 3D Effect */}
            <div className="absolute inset-0 opacity-30 pointer-events-none">
                <ThreeGlobe />
            </div>

            <div className="relative z-10 w-full max-w-md p-4">
                <GlassCard className="border-t-indigo-500/50">
                    <div className="text-center mb-8">
                        <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-cyan-400">
                            Welcome Back
                        </h1>
                        <p className="text-gray-400 mt-2">Access the AI Decision System</p>
                    </div>

                    <form onSubmit={handleLogin} className="space-y-6">
                        <div>
                            <label className="block text-sm font-medium text-gray-400 mb-2">Email</label>
                            <input
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                className="w-full bg-black/50 border border-gray-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-indigo-500 transition-colors"
                                placeholder="trader@example.com"
                                required
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-400 mb-2">Password</label>
                            <input
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                className="w-full bg-black/50 border border-gray-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-indigo-500 transition-colors"
                                placeholder="••••••••"
                                required
                            />
                        </div>

                        {error && (
                            <motion.p
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                className="text-red-500 text-sm text-center"
                            >
                                {error}
                            </motion.p>
                        )}

                        <NeonButton type="submit" className="w-full" isLoading={loading}>
                            Sign In
                        </NeonButton>

                        <div className="text-center text-sm text-gray-500 mt-4">
                            Don't have an account? <button type="button" onClick={() => router.push('/register')} className="text-indigo-400 hover:text-indigo-300 font-medium transition-colors">Create one</button>
                        </div>
                    </form>
                </GlassCard>
            </div>
        </div>
    )
}
