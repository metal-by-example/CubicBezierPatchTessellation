
import Foundation
import Metal
import MetalKit

struct VertexUniforms {
    var modelViewMatrix: simd_float4x4
    var projectionMatrix: simd_float4x4
}

class Renderer : NSObject, MTKViewDelegate {
    let view: MTKView
    let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    
    var cameraTransform = simd_float4x4.identity
    
    private var tessellationFactorPipeline: MTLComputePipelineState!
    private var controlPointPipeline: MTLComputePipelineState!
    private var renderPipelineState: MTLRenderPipelineState!
    
    private var depthStencilState: MTLDepthStencilState!

    private var tessellationFactorBuffer: MTLBuffer!
    private var controlPointPositionBuffer: MTLBuffer!
    
    init(view: MTKView) {
        self.view = view
        
        guard let device = view.device else {
            fatalError("MTKView must be configured with device before instantating Renderer")
        }
        
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        
        super.init()
        
        view.delegate = self
        
        makeResources()
        makePipelines()
    }
    
    func makeResources() {
        tessellationFactorBuffer = device.makeBuffer(length: MemoryLayout<MTLQuadTessellationFactorsHalf>.stride * 32,
                                                    options: .storageModePrivate)
        
        controlPointPositionBuffer = device.makeBuffer(length: MemoryLayout<float3>.stride * 512,
                                                       options: .storageModePrivate)
    }
    
    func makePipelines() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Unable to create default Metal library")
        }
        
        do {
            let tessFactorFunction = library.makeFunction(name: "compute_tess_factors")!
            tessellationFactorPipeline = try device.makeComputePipelineState(function: tessFactorFunction)
            
            let controlPointFunction = library.makeFunction(name: "compute_control_point_position")!
            controlPointPipeline = try device.makeComputePipelineState(function: controlPointFunction)

            let vertexFunction = library.makeFunction(name: "cub_bez_vertex")!
            let fragmentFunction = library.makeFunction(name: "fragment_main")!

            let renderPipelineDescriptor = MTLRenderPipelineDescriptor()
            renderPipelineDescriptor.rasterSampleCount = view.sampleCount
            
            renderPipelineDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
            renderPipelineDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat
            
            let vertexDescriptor = MTLVertexDescriptor()
            vertexDescriptor.attributes[0].format = .float3
            vertexDescriptor.attributes[0].offset = 0
            vertexDescriptor.attributes[0].bufferIndex = 0
            vertexDescriptor.layouts[0].stride = MemoryLayout<float3>.stride
            vertexDescriptor.layouts[0].stepFunction = .perPatchControlPoint
            vertexDescriptor.layouts[0].stepRate = 1
            renderPipelineDescriptor.vertexDescriptor = vertexDescriptor

            renderPipelineDescriptor.tessellationFactorFormat = .half
            renderPipelineDescriptor.tessellationOutputWindingOrder = .counterClockwise

            renderPipelineDescriptor.vertexFunction = vertexFunction
            renderPipelineDescriptor.fragmentFunction = fragmentFunction
            renderPipelineState = try device.makeRenderPipelineState(descriptor: renderPipelineDescriptor)
        } catch {
            print("Failed to create pipelines: \(error)")
        }
        
        let depthStencilDescriptor = MTLDepthStencilDescriptor()
        depthStencilDescriptor.depthCompareFunction = .less
        depthStencilDescriptor.isDepthWriteEnabled = true
        depthStencilState = device.makeDepthStencilState(descriptor: depthStencilDescriptor)
    }
    
    // MARK: - MTKViewDelegate
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
    }
    
    func draw(in view: MTKView) {
        guard let passDescriptor = view.currentRenderPassDescriptor else {
            return
        }
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        
        let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
        
        do {
            computeCommandEncoder.setBuffer(controlPointPositionBuffer, offset: 0, index: 0)
            computeCommandEncoder.setComputePipelineState(controlPointPipeline)
            let gridSize = MTLSize(width: 512, height: 1, depth: 1)
            let threadsPerThreadgroup = MTLSize(width: 16, height: 1, depth: 1)
            computeCommandEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadgroup)
        }

        do {
            computeCommandEncoder.setBuffer(tessellationFactorBuffer, offset: 0, index: 0)
            computeCommandEncoder.setComputePipelineState(tessellationFactorPipeline)
            let gridSize = MTLSize(width: 32, height: 1, depth: 1)
            let threadsPerThreadgroup = MTLSize(width: tessellationFactorPipeline.threadExecutionWidth,
                                                height: 1,
                                                depth: 1)
            computeCommandEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadgroup)
        }

        computeCommandEncoder.endEncoding()
        
        let renderCommandEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDescriptor)!
        
        renderCommandEncoder.setCullMode(.back)
        renderCommandEncoder.setFrontFacing(.counterClockwise)
        renderCommandEncoder.setDepthStencilState(depthStencilState)
        
        let zUpToYUp = simd_float4x4(columns: (float4(1, 0, 0, 0),
                                               float4(0, 0, 1, 0),
                                               float4(0, 1, 0, 0),
                                               float4(0, 0, 0, 1)))
        let modelMatrix = simd_float4x4(translation: float3(0, -1.0, 0)) * zUpToYUp
        
        let viewMatrix = cameraTransform.inverse
        
        let aspectRatio = Float(view.drawableSize.width / view.drawableSize.height)
        let projectionMatrix = float4x4(perspectiveFOV: rad_from_deg(65.0),
                                        aspectRatio: aspectRatio,
                                        nearZ: 0.1,
                                        farZ: 20.0)
        
        var uniforms = VertexUniforms(modelViewMatrix: viewMatrix * modelMatrix,
                                      projectionMatrix: projectionMatrix)
        
        renderCommandEncoder.setRenderPipelineState(renderPipelineState)
        
        renderCommandEncoder.setVertexBuffer(controlPointPositionBuffer, offset: 0, index: 0)
        renderCommandEncoder.setVertexBytes(&uniforms, length: MemoryLayout.size(ofValue: uniforms), index: 1)

        renderCommandEncoder.setTessellationFactorBuffer(tessellationFactorBuffer,
                                                         offset: 0,
                                                         instanceStride: 0)
        
        renderCommandEncoder.drawPatches(numberOfPatchControlPoints: 16,
                                         patchStart: 0,
                                         patchCount: 32,
                                         patchIndexBuffer: nil,
                                         patchIndexBufferOffset: 0,
                                         instanceCount: 1,
                                         baseInstance: 0)
        
        renderCommandEncoder.endEncoding()
    
        commandBuffer.present(view.currentDrawable!)
        commandBuffer.commit()
    }
}
