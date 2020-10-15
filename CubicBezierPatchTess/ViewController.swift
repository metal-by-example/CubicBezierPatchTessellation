
import Cocoa
import Metal
import MetalKit

class ViewController: NSViewController {
    
    var renderer: Renderer!
    
    private var previousMouseLocation = CGPoint.zero
    private var azimuth: Float = .pi / 2
    private var elevation: Float = .pi / 6
    private let cameraDistance: Float = 6.0
    private let rotationFactor: Float = 0.02

    override func viewDidLoad() {
        super.viewDidLoad()
        
        let device = MTLCreateSystemDefaultDevice()!
        
        // Add MetalKit view as subview of main view
        let mtkView = MTKView(frame: view.bounds, device: device)
        mtkView.autoresizingMask = [ .width, .height ]
        view.addSubview(mtkView)
        
        // Configure properties of MTKView that affect rendering
        mtkView.device = device
        mtkView.sampleCount = 4
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.depthStencilPixelFormat = .depth32Float
        
        renderer = Renderer(view: mtkView)
        updateCamera()
    }

    private func updateCamera() {
        let x = cos(azimuth) * cos(elevation)
        let y = sin(elevation)
        let z = sin(azimuth) * cos(elevation)
        let position = cameraDistance * float3(x, y, z)
        
        renderer.cameraTransform = float4x4(lookAt: float3(0, 0, 0),
                                            from: position,
                                            up: float3(0, 1, 0))
    }

    override func mouseDown(with event: NSEvent) {
        previousMouseLocation = view.convert(event.locationInWindow, from: nil)
    }
    
    override func mouseDragged(with event: NSEvent) {
        let mouseLocation = view.convert(event.locationInWindow, from: nil)
        let deltaX = Float(mouseLocation.x - previousMouseLocation.x) * rotationFactor
        let deltaY = Float(mouseLocation.y - previousMouseLocation.y) * rotationFactor

        azimuth += deltaX
        elevation += -deltaY
        elevation = max(-.pi / 2, min(elevation, .pi / 2))
        updateCamera()

        previousMouseLocation = mouseLocation
    }

}
