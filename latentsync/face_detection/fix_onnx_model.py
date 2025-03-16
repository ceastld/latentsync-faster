import os
import onnx
from onnx import helper, shape_inference
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

def remove_unused_initializers(model_path):
    """
    Remove unused initializers and batch norm tracking variables from the ONNX model.
    
    Args:
        model_path: Path to the ONNX model file
    """
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    print(f"Processing model: {model_path}")
    
    # Load the model
    model = onnx.load(model_path)
    
    # Get list of initializer names
    initializer_names = {x.name for x in model.graph.initializer}
    
    # Get list of all node inputs
    node_inputs = set()
    for node in model.graph.node:
        node_inputs.update(node.input)
    
    # Remove initializers from inputs and unused initializers
    inputs = []
    new_initializers = []
    removed_count = 0
    
    # Filter graph inputs
    for input in model.graph.input:
        if input.name not in initializer_names:
            inputs.append(input)
    
    # Filter initializers
    for initializer in model.graph.initializer:
        # Remove if it's a num_batches_tracked variable or not used by any node
        if "num_batches_tracked" in initializer.name or initializer.name not in node_inputs:
            removed_count += 1
            print(f"Removing unused initializer: {initializer.name}")
        else:
            new_initializers.append(initializer)
    
    # Clear and update the graph
    model.graph.ClearField('input')
    model.graph.input.extend(inputs)
    
    model.graph.ClearField('initializer')
    model.graph.initializer.extend(new_initializers)
    
    # Perform shape inference
    model = shape_inference.infer_shapes(model)
    
    # Save the modified model
    output_path = model_path.replace('.onnx', '_fixed.onnx')
    onnx.save(model, output_path)
    print(f"Fixed model saved to: {output_path}")
    print(f"Removed {removed_count} unused initializers")

if __name__ == "__main__":
    # Fix both models
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    face_detector_path = os.path.join(models_dir, "face_detector.onnx")
    landmark_detector_path = os.path.join(models_dir, "landmark_detector.onnx")
    
    remove_unused_initializers(face_detector_path)
    remove_unused_initializers(landmark_detector_path) 