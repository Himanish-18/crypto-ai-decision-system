import { useEffect, useState } from 'react'
import Head from 'next/head'
import { useRouter } from 'next/router'
import { motion, AnimatePresence } from 'framer-motion'
import { NeonButton } from '@/components/NeonButton'
import { GlassCard } from '@/components/GlassCard'
import dynamic from 'next/dynamic'
import api from '@/services/api'

const ParticleBackground = dynamic(() => import('@/components/ParticleBackground'), { ssr: false })

interface Trade {
    id: number
    symbol: string
    side: string
    price: number
    amount: number
    pnl: number | null
    timestamp: string
}

export default function Dashboard() {
    const router = useRouter()
    const [price, setPrice] = useState(42000.00)
    const [trades, setTrades] = useState<Trade[]>([])
    const [amount, setAmount] = useState('')
    const [loading, setLoading] = useState(false)
    const [ws, setWs] = useState<WebSocket | null>(null)

    const fetchTrades = async () => {
        try {
            const { data } = await api.get('/trades')
            setTrades(data)
        } catch (err) {
            console.error('Failed to fetch trades')
        }
    }

    useEffect(() => {
        const token = localStorage.getItem('token')
        if (!token) router.push('/login')
        else {
            // Set auth header
            api.defaults.headers.common['Authorization'] = `Bearer ${token}`
            fetchTrades()
        }

        const socket = new WebSocket('ws://localhost:8000/ws')
        socket.onmessage = (event) => {
            const data = JSON.parse(event.data)
            if (data.type === 'market_data') {
                setPrice(parseFloat(data.data.k.c))
            }
        }
        setWs(socket)

        return () => socket.close()
    }, [])

    const handleTrade = async (side: 'BUY' | 'SELL') => {
        if (!amount) return
        setLoading(true)
        try {
            await api.post('/trades', {
                symbol: 'BTC/USDT',
                side,
                price,
                amount: parseFloat(amount)
            })
            setAmount('')
            await fetchTrades()
        } catch (err) {
            alert('Trade failed')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen bg-black text-white p-4 relative overflow-hidden">
            <Head>
                <title>Terminal | Crypto AI</title>
            </Head>

            <div className="absolute inset-0 opacity-30 pointer-events-none">
                <ParticleBackground />
            </div>

            <div className="relative z-10 max-w-[1920px] mx-auto">
                {/* Header */}
                <header className="flex justify-between items-center mb-6 px-4 py-2 bg-gray-900/50 backdrop-blur rounded-xl border border-white/10">
                    <div className="flex items-center gap-4">
                        <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-cyan-400">
                            ANTIGRAVITY TERMINAL
                        </h1>
                        <div className="flex items-center gap-2 px-3 py-1 bg-green-500/10 text-green-400 rounded-full text-xs font-mono border border-green-500/20">
                            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                            SYSTEM ONLINE
                        </div>
                    </div>
                    <div className="flex items-center gap-6">
                        <div className="text-right">
                            <p className="text-xs text-gray-400 uppercase tracking-wider">BTC/USDT</p>
                            <p className="text-xl font-mono font-bold text-white">${price.toFixed(2)}</p>
                        </div>
                        <NeonButton variant="danger" onClick={() => {
                            localStorage.removeItem('token')
                            router.push('/login')
                        }} className="px-4 py-2 text-sm">Logout</NeonButton>
                    </div>
                </header>

                {/* Main Grid */}
                <div className="grid grid-cols-12 gap-6 h-[calc(100vh-120px)]">

                    {/* Left: Chart & Positions */}
                    <div className="col-span-9 flex flex-col gap-6">
                        <GlassCard className="flex-1 relative overflow-hidden p-0 border-indigo-500/20">
                            <div className="absolute top-4 left-4 z-10 flex gap-2">
                                {['1H', '4H', '1D'].map(tf => (
                                    <button key={tf} className="px-3 py-1 bg-black/50 hover:bg-gray-700 rounded text-xs transition border border-white/10">
                                        {tf}
                                    </button>
                                ))}
                            </div>
                            <div className="w-full h-full flex items-center justify-center bg-gradient-to-b from-transparent to-indigo-900/5">
                                <div className="text-center">
                                    <p className="text-gray-500 font-mono animate-pulse mb-2">INITIALIZING NEURAL CHART ENGINE...</p>
                                    <div className="w-64 h-1 bg-gray-800 rounded-full overflow-hidden mx-auto">
                                        <div className="w-1/2 h-full bg-indigo-500 animate-[shimmer_2s_infinite]" />
                                    </div>
                                </div>
                            </div>
                        </GlassCard>

                        {/* Bottom: Positions */}
                        <GlassCard className="h-72 overflow-hidden flex flex-col">
                            <div className="flex justify-between items-center mb-4">
                                <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider">Trade History</h3>
                                <span className="text-xs text-gray-500 font-mono">SYNCED</span>
                            </div>
                            <div className="overflow-auto flex-1">
                                <table className="w-full text-sm text-left">
                                    <thead className="text-gray-500 border-b border-gray-800 sticky top-0 bg-gray-900/90 backdrop-blur">
                                        <tr>
                                            <th className="pb-2 pl-2">Time</th>
                                            <th className="pb-2">Symbol</th>
                                            <th className="pb-2">Side</th>
                                            <th className="pb-2 text-right">Size</th>
                                            <th className="pb-2 text-right">Price</th>
                                            <th className="pb-2 text-right pr-2">Status</th>
                                        </tr>
                                    </thead>
                                    <tbody className="font-mono">
                                        <AnimatePresence>
                                            {trades.map((trade) => (
                                                <motion.tr
                                                    key={trade.id}
                                                    initial={{ opacity: 0, x: -20 }}
                                                    animate={{ opacity: 1, x: 0 }}
                                                    className="border-b border-gray-800/50 hover:bg-white/5 transition group"
                                                >
                                                    <td className="py-3 pl-2 text-gray-400">{new Date(trade.timestamp).toLocaleTimeString()}</td>
                                                    <td className="py-3 text-white font-bold">{trade.symbol}</td>
                                                    <td className={`py-3 font-bold ${trade.side === 'BUY' ? 'text-green-400' : 'text-red-400'}`}>
                                                        {trade.side}
                                                    </td>
                                                    <td className="py-3 text-right">{trade.amount}</td>
                                                    <td className="py-3 text-right">${trade.price.toFixed(2)}</td>
                                                    <td className="py-3 text-right pr-2 text-indigo-400">FILLED</td>
                                                </motion.tr>
                                            ))}
                                        </AnimatePresence>
                                    </tbody>
                                </table>
                                {trades.length === 0 && (
                                    <div className="text-center py-8 text-gray-600">No trades executed yet.</div>
                                )}
                            </div>
                        </GlassCard>
                    </div>

                    {/* Right: Order Panel */}
                    <div className="col-span-3 flex flex-col gap-6">
                        <GlassCard className="flex-1 border-t-4 border-t-indigo-500" floating>
                            <h3 className="text-sm font-bold text-gray-400 mb-6 uppercase tracking-wider flex items-center gap-2">
                                <div className="w-2 h-2 bg-indigo-500 rounded-full" />
                                Order Entry
                            </h3>

                            <div className="space-y-6">
                                <div>
                                    <label className="text-xs text-gray-500 uppercase tracking-wider mb-1 block">Market Price</label>
                                    <div className="bg-black/40 border border-gray-700 rounded-lg px-4 py-3 font-mono text-2xl text-right text-white shadow-inner">
                                        ${price.toFixed(2)}
                                    </div>
                                </div>

                                <div>
                                    <label className="text-xs text-gray-500 uppercase tracking-wider mb-1 block">Amount (BTC)</label>
                                    <input
                                        type="number"
                                        value={amount}
                                        onChange={(e) => setAmount(e.target.value)}
                                        className="w-full bg-black/40 border border-gray-700 rounded-lg px-4 py-3 font-mono text-right text-white focus:border-indigo-500 outline-none transition shadow-inner"
                                        placeholder="0.00"
                                    />
                                </div>

                                <div className="grid grid-cols-2 gap-3 pt-4">
                                    <NeonButton
                                        variant="success"
                                        className="w-full py-4 text-lg"
                                        onClick={() => handleTrade('BUY')}
                                        isLoading={loading}
                                    >
                                        BUY
                                    </NeonButton>
                                    <NeonButton
                                        variant="danger"
                                        className="w-full py-4 text-lg"
                                        onClick={() => handleTrade('SELL')}
                                        isLoading={loading}
                                    >
                                        SELL
                                    </NeonButton>
                                </div>

                                <div className="pt-4 border-t border-gray-800">
                                    <div className="flex justify-between text-sm text-gray-400">
                                        <span>Est. Total</span>
                                        <span className="font-mono text-white">${(price * (parseFloat(amount) || 0)).toFixed(2)}</span>
                                    </div>
                                </div>
                            </div>
                        </GlassCard>

                        <GlassCard className="h-1/3 border-t-4 border-t-green-500" floating>
                            <h3 className="text-sm font-bold text-gray-400 mb-4 uppercase tracking-wider flex items-center gap-2">
                                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                                AI Strategy
                            </h3>
                            <div className="flex items-center justify-between mb-6 p-4 bg-green-500/10 rounded-lg border border-green-500/20">
                                <span className="text-2xl font-bold text-green-400">LONG</span>
                                <div className="text-right">
                                    <p className="text-xs text-green-300/70 uppercase">Confidence</p>
                                    <p className="text-xl font-bold text-green-400">94.2%</p>
                                </div>
                            </div>
                            <div className="space-y-2">
                                <div className="flex justify-between text-xs">
                                    <span className="text-gray-500">Regime</span>
                                    <span className="text-indigo-400 font-bold">TRENDING_UP</span>
                                </div>
                                <div className="flex justify-between text-xs">
                                    <span className="text-gray-500">Volatility</span>
                                    <span className="text-gray-300">LOW (1.2%)</span>
                                </div>
                                <div className="w-full bg-gray-800 h-1 rounded-full mt-2 overflow-hidden">
                                    <div className="w-[94%] h-full bg-green-500" />
                                </div>
                            </div>
                        </GlassCard>
                    </div>
                </div>
            </div>
        </div>
    )
}
