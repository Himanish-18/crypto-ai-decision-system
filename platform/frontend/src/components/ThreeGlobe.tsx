import { useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Sphere, MeshDistortMaterial } from '@react-three/drei'

function AnimatedGlobe() {
    const meshRef = useRef<any>(null)

    useFrame((state) => {
        if (meshRef.current) {
            meshRef.current.rotation.x = state.clock.getElapsedTime() * 0.2
            meshRef.current.rotation.y = state.clock.getElapsedTime() * 0.3
        }
    })

    return (
        <Sphere args={[1, 32, 32]} ref={meshRef} scale={2}>
            <MeshDistortMaterial
                color="#4f46e5"
                attach="material"
                distort={0.5}
                speed={2}
                roughness={0}
            />
        </Sphere>
    )
}

export default function ThreeGlobe() {
    return (
        <div className="h-[400px] w-full">
            <Canvas>
                <ambientLight intensity={0.5} />
                <directionalLight position={[10, 10, 5]} intensity={1} />
                <AnimatedGlobe />
            </Canvas>
        </div>
    )
}
