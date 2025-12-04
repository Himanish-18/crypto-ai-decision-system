import Head from 'next/head'
import { useRouter } from 'next/router'
import dynamic from 'next/dynamic'
import { NeonButton } from '@/components/NeonButton'
import { GlassCard } from '@/components/GlassCard'

const ThreeGlobe = dynamic(() => import('@/components/ThreeGlobe'), { ssr: false })

export default function Home() {
    const router = useRouter()

    return (
        <div className="min-h-screen bg-black text-white overflow-hidden relative">
            <Head>
                <title>Crypto AI | Institutional Grade</title>
            </Head>

            {/* 3D Background */}
            <div className="absolute inset-0 opacity-40">
                <ThreeGlobe />
            </div>

            <main className="relative z-10 max-w-7xl mx-auto px-6 pt-32">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">

                    <div className="space-y-8">
                        <div className="inline-block px-4 py-2 bg-indigo-500/10 border border-indigo-500/30 rounded-full">
                            <span className="text-indigo-400 font-mono text-sm">V2.0 SYSTEM ONLINE</span>
                        </div>

                        <h1 className="text-6xl font-bold leading-tight">
                            The Future of <br />
                            <span className="bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-purple-400 to-cyan-400">
                                Algorithmic Trading
                            </span>
                        </h1>

                        <p className="text-xl text-gray-400 max-w-lg leading-relaxed">
                            Institutional-grade AI decision system.
                            Real-time regime detection, reinforcement learning execution, and
                            sub-millisecond latency.
                        </p>

                        <div className="flex gap-4">
                            <NeonButton onClick={() => router.push('/login')} className="px-8 py-4 text-lg">
                                Launch Terminal
                            </NeonButton>
                            <NeonButton variant="success" className="px-8 py-4 text-lg bg-transparent border border-white/20 hover:bg-white/5">
                                View Documentation
                            </NeonButton>
                        </div>

                        <div className="grid grid-cols-3 gap-8 pt-12 border-t border-white/10">
                            <div>
                                <p className="text-3xl font-bold font-mono">$42B+</p>
                                <p className="text-sm text-gray-500 uppercase tracking-wider mt-1">Volume Processed</p>
                            </div>
                            <div>
                                <p className="text-3xl font-bold font-mono text-green-400">99.9%</p>
                                <p className="text-sm text-gray-500 uppercase tracking-wider mt-1">Uptime</p>
                            </div>
                            <div>
                                <p className="text-3xl font-bold font-mono text-indigo-400">50ms</p>
                                <p className="text-sm text-gray-500 uppercase tracking-wider mt-1">Execution Speed</p>
                            </div>
                        </div>
                    </div>

                    <div className="relative">
                        <GlassCard className="p-8 border-t-indigo-500/50">
                            <div className="flex justify-between items-center mb-6">
                                <h3 className="font-bold text-gray-400 uppercase tracking-wider">Live Signal</h3>
                                <div className="flex gap-2">
                                    <div className="w-3 h-3 rounded-full bg-red-500/20" />
                                    <div className="w-3 h-3 rounded-full bg-yellow-500/20" />
                                    <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse" />
                                </div>
                            </div>

                            <div className="space-y-6">
                                <div className="flex justify-between items-end">
                                    <div>
                                        <p className="text-sm text-gray-500 mb-1">Asset</p>
                                        <p className="text-2xl font-bold">BTC/USDT</p>
                                    </div>
                                    <div className="text-right">
                                        <p className="text-sm text-gray-500 mb-1">Price</p>
                                        <p className="text-2xl font-mono">$42,150.00</p>
                                    </div>
                                </div>

                                <div className="h-32 bg-gradient-to-b from-indigo-500/10 to-transparent rounded-lg border border-indigo-500/20 flex items-center justify-center relative overflow-hidden">
                                    <div className="absolute inset-0 flex items-center justify-center">
                                        <span className="text-6xl font-bold text-white/5">AI</span>
                                    </div>
                                    <div className="text-center z-10">
                                        <p className="text-green-400 font-bold text-xl mb-1">STRONG BUY</p>
                                        <p className="text-xs text-gray-400">Confidence: 94.2%</p>
                                    </div>
                                </div>

                                <div className="grid grid-cols-2 gap-4">
                                    <div className="bg-black/40 rounded p-3">
                                        <p className="text-xs text-gray-500">Volatility</p>
                                        <p className="font-mono text-indigo-400">LOW</p>
                                    </div>
                                    <div className="bg-black/40 rounded p-3">
                                        <p className="text-xs text-gray-500">Trend</p>
                                        <p className="font-mono text-green-400">UPWARD</p>
                                    </div>
                                </div>
                            </div>
                        </GlassCard>
                    </div>

                </div>
            </main>
        </div>
    )
}
