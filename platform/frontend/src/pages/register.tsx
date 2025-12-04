import { useState } from 'react'
import { useRouter } from 'next/router'
import Head from 'next/head'
import dynamic from 'next/dynamic'
import { motion } from 'framer-motion'
import { NeonButton } from '@/components/NeonButton'
import { GlassCard } from '@/components/GlassCard'
import { auth } from '@/services/api'

const ParticleBackground = dynamic(() => import('@/components/ParticleBackground'), { ssr: false })

export default function Register() {
    const router = useRouter()
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    const handleRegister = async (e: React.FormEvent) => {
        e.preventDefault()
        setLoading(true)
        setError('')
        try {
            await auth.register(email, password)
            // Auto login after register
            const { access_token } = await auth.login(email, password)
            localStorage.setItem('token', access_token)
            router.push('/dashboard')
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Registration failed')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen bg-black flex items-center justify-center relative overflow-hidden">
            <Head>
                <title>Register | Crypto AI</title>
            </Head>

            <div className="absolute inset-0 opacity-50 pointer-events-none">
                <ParticleBackground />
            </div>

            <div className="relative z-10 w-full max-w-md p-4">
                <GlassCard className="border-t-purple-500/50" floating>
                    <div className="text-center mb-8">
                        <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-400">
                            Create Account
                        </h1>
                        <p className="text-gray-400 mt-2">Join the Institutional Network</p>
                    </div>

                    <form onSubmit={handleRegister} className="space-y-6">
                        <div>
                            <label className="block text-sm font-medium text-gray-400 mb-2">Email</label>
                            <input
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                className="w-full bg-black/50 border border-gray-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-purple-500 transition-colors"
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
                                className="w-full bg-black/50 border border-gray-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-purple-500 transition-colors"
                                placeholder="••••••••"
                                required
                            />
                        </div>

                        {error && (
                            <p className="text-red-500 text-sm text-center">{error}</p>
                        )}

                        <NeonButton type="submit" className="w-full bg-purple-600 hover:bg-purple-500 border-purple-400/30 shadow-[0_0_15px_rgba(147,51,234,0.5)]" isLoading={loading}>
                            Initialize Access
                        </NeonButton>

                        <div className="text-center text-sm text-gray-500">
                            Already have access? <button type="button" onClick={() => router.push('/login')} className="text-purple-400 hover:text-purple-300">Login</button>
                        </div>
                    </form>
                </GlassCard>
            </div>
        </div>
    )
}
