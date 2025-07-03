#!/bin/bash

# GPU Quantization Test Runner
# Run this script on your GPU server to test quantization capabilities

echo "🚀 Starting GPU Quantization Tests"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "test_gpu_quantization_offline.py" ]; then
    echo "❌ Error: test_gpu_quantization_offline.py not found"
    echo "Please run this script from the ModelConverterTool directory"
    exit 1
fi

# Check Python environment
echo "📋 Checking Python environment..."
python --version
pip list | grep -E "(torch|transformers|auto-gptq|gptqmodel)"

# Test 1: Offline GPU Quantization Test
echo ""
echo "🧪 Test 1: Offline GPU Quantization Test"
echo "========================================"
python test_gpu_quantization_offline.py

if [ $? -eq 0 ]; then
    echo "✅ Offline test passed!"
else
    echo "❌ Offline test failed"
fi

# Test 2: Comprehensive GPU Quantization Test (if network available)
echo ""
echo "🧪 Test 2: Comprehensive GPU Quantization Test"
echo "=============================================="
echo "Note: This test requires network access to HuggingFace Hub"
echo "If network is not available, this test will fail gracefully"

python test_gpu_quantization_comprehensive.py

if [ $? -eq 0 ]; then
    echo "✅ Comprehensive test passed!"
else
    echo "⚠️ Comprehensive test failed (expected if no network access)"
fi

# Test 3: Basic GPU Functionality Test
echo ""
echo "🧪 Test 3: Basic GPU Functionality Test"
echo "======================================"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    
    # Test basic GPU operations
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = torch.matmul(x, y)
    print(f'GPU tensor operation successful: {z.shape}')
    print('✅ Basic GPU functionality confirmed!')
else:
    print('❌ CUDA not available')
"

echo ""
echo "📊 Test Summary"
echo "==============="
echo "Check the log files for detailed results:"
echo "- gpu_quantization_test.log (comprehensive test)"
echo "- Console output above"

echo ""
echo "🎯 Next Steps:"
echo "1. If offline test passed: Your GPU quantization setup is working!"
echo "2. If you have network access: Try the comprehensive test"
echo "3. If tests failed: Check the error messages and dependencies" 