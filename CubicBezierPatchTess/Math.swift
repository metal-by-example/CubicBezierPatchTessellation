
import Foundation
import simd

func rad_from_deg(_ degrees: Float) -> Float {
    return (degrees / 180) * .pi
}

typealias float2 = SIMD2<Float>
typealias float3 = SIMD3<Float>
typealias float4 = SIMD4<Float>

extension float4x4 {
    
    static var identity = matrix_identity_float4x4
    
    init(rotationAbout axis: SIMD3<Float>, by angleRadians: Float) {
        let unitAxis = normalize(axis)
        let ct = cosf(angleRadians)
        let st = sinf(angleRadians)
        let ci = 1 - ct
        let x = unitAxis.x, y = unitAxis.y, z = unitAxis.z
        self.init(columns:(float4(    ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
                           float4(x * y * ci - z * st,     ct + y * y * ci, z * y * ci + x * st, 0),
                           float4(x * z * ci + y * st, y * z * ci - x * st,     ct + z * z * ci, 0),
                           float4(                  0,                   0,                   0, 1)))
    }
    
    init(translation t: float3) {
        self.init(columns:(float4(1, 0, 0, 0),
                           float4(0, 1, 0, 0),
                           float4(0, 0, 1, 0),
                           float4(t.x, t.y, t.z, 1)))
    }
    
    init (perspectiveFOV fovy: Float, aspectRatio: Float, nearZ: Float, farZ: Float) {
        let ys = 1 / tanf(fovy * 0.5)
        let xs = ys / aspectRatio
        let zs = farZ / (nearZ - farZ)
        self.init(columns:(float4(xs,  0, 0,   0),
                           float4( 0, ys, 0,   0),
                           float4( 0,  0, zs, -1),
                           float4( 0,  0, zs * nearZ, 0)))
    }
    
    init(lookAt to: float3, from: float3, up: float3) {
        let nz = normalize(to - from)
        let x = normalize(cross(nz, up))
        let y = normalize(cross(x, nz))
        self.init([float4(   x, 0.0),
                   float4(   y, 0.0),
                   float4( -nz, 0.0),
                   float4(from, 1.0)])
    }
}
