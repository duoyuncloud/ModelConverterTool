#!/usr/bin/env python3
"""
Comprehensive test to verify the authenticity and correctness of GPT-2 conversions
"""

import tempfile
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from model_converter_tool.converter import ModelConverter
from model_converter_tool.validator import ModelValidator


def test_conversion_authenticity():
    """Test the authenticity and correctness of GPT-2 conversions"""
    
    print("🔍 GPT-2 转换真实性和正确性验证测试")
    print("=" * 60)
    
    converter = ModelConverter()
    validator = ModelValidator()
    
    # 测试格式列表
    test_formats = [
        ('fp16', 'FP16 半精度'),
        ('torchscript', 'TorchScript'),
        ('hf', 'Hugging Face'),
        ('onnx', 'ONNX'),
    ]
    
    results = {}
    
    for format_type, format_name in test_formats:
        print(f"\n📋 测试 {format_name} 转换")
        print("-" * 40)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / f'gpt2_{format_type}'
            out_dir.mkdir()
            
            # 记录转换开始时间
            start_time = time.time()
            
            # 执行转换
            result = converter.convert(
                input_source='hf:gpt2',
                output_format=format_type,
                output_path=str(out_dir),
                model_type='text-generation',
                validate=True
            )
            
            conversion_time = time.time() - start_time
            
            print(f"转换成功: {'✅' if result['success'] else '❌'}")
            print(f"转换时间: {conversion_time:.2f}秒")
            print(f"验证成功: {'✅' if result.get('validation', False) else '❌'}")
            
            if result['success']:
                # 文件分析
                files = list(out_dir.glob('*'))
                print(f"生成文件数量: {len(files)}")
                
                # 检查关键文件
                key_files = get_key_files_for_format(format_type)
                file_analysis = analyze_files(out_dir, key_files)
                
                for file_info in file_analysis:
                    status = "✅" if file_info['exists'] else "❌"
                    size_str = f"{file_info['size']:.2f} MB" if file_info['exists'] else "缺失"
                    print(f"  {status} {file_info['name']}: {size_str}")
                
                # 模型验证分析
                model_validation = result.get('model_validation', {})
                if model_validation:
                    print(f"模型验证: {'✅' if model_validation.get('success', False) else '❌'}")
                    print(f"验证消息: {model_validation.get('message', 'N/A')}")
                
                # 实际推理测试
                inference_test = test_actual_inference(out_dir, format_type)
                print(f"推理测试: {'✅' if inference_test['success'] else '❌'}")
                if inference_test['success']:
                    print(f"  输入: \"{inference_test['input_text']}\"")
                    print(f"  输出: \"{inference_test['output_text']}\"")
                    print(f"  输出形状: {inference_test['output_shape']}")
                
                # 保存结果
                results[format_type] = {
                    'success': True,
                    'conversion_time': conversion_time,
                    'validation': result.get('validation', False),
                    'file_count': len(files),
                    'key_files': file_analysis,
                    'model_validation': model_validation,
                    'inference_test': inference_test
                }
            else:
                print(f"转换失败: {result.get('error', '未知错误')}")
                results[format_type] = {
                    'success': False,
                    'error': result.get('error', '未知错误')
                }
    
    # 生成总结报告
    print("\n" + "=" * 60)
    print("📊 转换真实性和正确性总结报告")
    print("=" * 60)
    
    successful_formats = []
    failed_formats = []
    
    for format_type, format_name in test_formats:
        result = results[format_type]
        if result['success']:
            successful_formats.append(format_type)
            print(f"\n✅ {format_name}:")
            print(f"  转换时间: {result['conversion_time']:.2f}秒")
            print(f"  文件数量: {result['file_count']}")
            print(f"  验证通过: {'是' if result['validation'] else '否'}")
            print(f"  推理测试: {'通过' if result['inference_test']['success'] else '失败'}")
            
            # 检查转换真实性
            authenticity_score = calculate_authenticity_score(result)
            print(f"  真实性评分: {authenticity_score:.1f}/10")
            
        else:
            failed_formats.append(format_type)
            print(f"\n❌ {format_name}: 转换失败")
            print(f"  错误: {result['error']}")
    
    print(f"\n📈 总体统计:")
    print(f"  成功格式: {len(successful_formats)}/{len(test_formats)}")
    print(f"  失败格式: {len(failed_formats)}/{len(test_formats)}")
    
    if successful_formats:
        avg_time = sum(results[f]['conversion_time'] for f in successful_formats) / len(successful_formats)
        print(f"  平均转换时间: {avg_time:.2f}秒")
    
    # 真实性评估
    print(f"\n🔍 真实性评估:")
    print(f"  FP16: {'真实转换' if 'fp16' in successful_formats else '转换失败'}")
    print(f"  TorchScript: {'真实转换' if 'torchscript' in successful_formats else '转换失败'}")
    print(f"  HF: {'真实转换' if 'hf' in successful_formats else '转换失败'}")
    print(f"  ONNX: {'部分转换' if 'onnx' in successful_formats else '转换失败'}")
    
    return results


def get_key_files_for_format(format_type: str) -> List[str]:
    """获取每种格式的关键文件列表"""
    key_files_map = {
        'fp16': ['model.safetensors', 'config.json', 'tokenizer.json'],
        'torchscript': ['model.pt', 'config.json', 'tokenizer.json'],
        'hf': ['pytorch_model.bin', 'config.json', 'tokenizer.json'],
        'onnx': ['model.onnx', 'config.json', 'tokenizer.json'],
    }
    return key_files_map.get(format_type, [])


def analyze_files(output_dir: Path, key_files: List[str]) -> List[Dict[str, Any]]:
    """分析输出文件"""
    analysis = []
    
    for file_name in key_files:
        file_path = output_dir / file_name
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)  # MB
            analysis.append({
                'name': file_name,
                'exists': True,
                'size': size,
                'path': str(file_path)
            })
        else:
            analysis.append({
                'name': file_name,
                'exists': False,
                'size': 0,
                'path': None
            })
    
    return analysis


def test_actual_inference(output_dir: Path, format_type: str) -> Dict[str, Any]:
    """测试实际推理能力"""
    try:
        if format_type == 'fp16':
            return test_fp16_inference(output_dir)
        elif format_type == 'torchscript':
            return test_torchscript_inference(output_dir)
        elif format_type == 'hf':
            return test_hf_inference(output_dir)
        elif format_type == 'onnx':
            return test_onnx_inference(output_dir)
        else:
            return {'success': False, 'error': f'Unsupported format: {format_type}'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def test_fp16_inference(output_dir: Path) -> Dict[str, Any]:
    """测试FP16模型推理"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model = AutoModelForCausalLM.from_pretrained(str(output_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
        
        # 测试推理
        test_text = "Hello world"
        inputs = tokenizer(test_text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 生成文本
        generated = model.generate(
            inputs['input_ids'], 
            max_length=len(inputs['input_ids'][0]) + 5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        
        return {
            'success': True,
            'input_text': test_text,
            'output_text': generated_text,
            'output_shape': outputs.logits.shape,
            'vocab_size': outputs.logits.shape[-1]
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def test_torchscript_inference(output_dir: Path) -> Dict[str, Any]:
    """测试TorchScript模型推理"""
    try:
        import torch
        
        model_path = output_dir / 'model.pt'
        if not model_path.exists():
            return {'success': False, 'error': 'TorchScript model file not found'}
        
        model = torch.jit.load(str(model_path))
        
        # 测试推理
        dummy_input = torch.randint(0, 50257, (1, 5))
        output = model(dummy_input)
        
        return {
            'success': True,
            'input_text': f"Token IDs: {dummy_input[0].tolist()}",
            'output_text': f"Output shape: {output.shape}",
            'output_shape': output.shape
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def test_hf_inference(output_dir: Path) -> Dict[str, Any]:
    """测试HF模型推理"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model = AutoModelForCausalLM.from_pretrained(str(output_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
        
        # 测试推理
        test_text = "Hello world"
        inputs = tokenizer(test_text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 生成文本
        generated = model.generate(
            inputs['input_ids'], 
            max_length=len(inputs['input_ids'][0]) + 5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        
        return {
            'success': True,
            'input_text': test_text,
            'output_text': generated_text,
            'output_shape': outputs.logits.shape,
            'vocab_size': outputs.logits.shape[-1]
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def test_onnx_inference(output_dir: Path) -> Dict[str, Any]:
    """测试ONNX模型推理"""
    try:
        import onnxruntime as ort
        import numpy as np
        
        onnx_files = list(output_dir.glob('*.onnx'))
        if not onnx_files:
            return {'success': False, 'error': 'No ONNX files found'}
        
        onnx_file = onnx_files[0]
        
        # 检查ONNX文件大小
        file_size = onnx_file.stat().st_size
        if file_size < 1024:  # 小于1KB，可能是占位符文件
            return {'success': False, 'error': 'ONNX file too small, likely a placeholder'}
        
        # 尝试加载ONNX模型
        try:
            session = ort.InferenceSession(str(onnx_file))
        except Exception as e:
            return {'success': False, 'error': f'ONNX loading failed: {e}'}
        
        # 获取输入输出信息
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # 创建测试输入
        dummy_input = np.random.randint(0, 50257, (1, 10), dtype=np.int64)
        
        # 运行推理
        outputs = session.run([output_name], {input_name: dummy_input})
        
        return {
            'success': True,
            'input_text': f"Token IDs: {dummy_input[0].tolist()[:5]}...",
            'output_text': f"Output shape: {outputs[0].shape}",
            'output_shape': outputs[0].shape
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def calculate_authenticity_score(result: Dict[str, Any]) -> float:
    """计算转换真实性评分 (0-10)"""
    score = 0.0
    
    if not result['success']:
        return 0.0
    
    # 基础分数：转换成功
    score += 2.0
    
    # 文件完整性
    key_files = result['key_files']
    existing_files = sum(1 for f in key_files if f['exists'])
    if key_files:
        file_completeness = existing_files / len(key_files)
        score += file_completeness * 2.0
    
    # 文件大小合理性
    total_size = sum(f['size'] for f in key_files if f['exists'])
    if total_size > 10:  # 大于10MB认为是合理的模型大小
        score += 2.0
    elif total_size > 1:  # 大于1MB
        score += 1.0
    
    # 模型验证
    if result.get('validation'):
        score += 1.0
    
    # 推理测试
    if result.get('inference_test', {}).get('success'):
        score += 2.0
    
    # 转换时间合理性
    conversion_time = result.get('conversion_time', 0)
    if 1 <= conversion_time <= 300:  # 1秒到5分钟之间
        score += 1.0
    
    return min(score, 10.0)


if __name__ == "__main__":
    results = test_conversion_authenticity()
    
    # 保存详细结果
    with open('conversion_authenticity_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 详细结果已保存到: conversion_authenticity_results.json") 