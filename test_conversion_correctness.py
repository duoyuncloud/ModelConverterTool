#!/usr/bin/env python3
"""
Deep test to verify the correctness of GPT-2 conversions by comparing weights and outputs
"""

import tempfile
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch

from model_converter_tool.converter import ModelConverter


def test_conversion_correctness():
    """Test the correctness of GPT-2 conversions by comparing with original model"""
    
    print("🔬 GPT-2 转换正确性深度验证测试")
    print("=" * 60)
    
    converter = ModelConverter()
    
    # 加载原始GPT-2模型作为基准
    print("📥 加载原始GPT-2模型作为基准...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    original_model = AutoModelForCausalLM.from_pretrained('gpt2')
    original_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # 获取原始模型的权重信息
    original_weights = {}
    for name, param in original_model.named_parameters():
        original_weights[name] = {
            'shape': tuple(param.shape),
            'dtype': str(param.dtype),
            'mean': float(param.mean().item()),
            'std': float(param.std().item()),
            'min': float(param.min().item()),
            'max': float(param.max().item())
        }
    
    print(f"原始模型参数数量: {len(original_weights)}")
    print(f"原始模型总参数量: {sum(p.numel() for p in original_model.parameters()):,}")
    
    # 测试输入
    test_text = "Hello world"
    test_inputs = original_tokenizer(test_text, return_tensors='pt')
    
    # 获取原始模型的输出
    with torch.no_grad():
        original_outputs = original_model(**test_inputs)
    original_logits = original_outputs.logits
    
    print(f"原始模型输出形状: {original_logits.shape}")
    print(f"原始模型输出统计: mean={original_logits.mean().item():.6f}, std={original_logits.std().item():.6f}")
    
    # 测试格式列表
    test_formats = [
        ('fp16', 'FP16 半精度'),
        ('hf', 'Hugging Face'),
    ]
    
    results = {}
    
    for format_type, format_name in test_formats:
        print(f"\n📋 测试 {format_name} 转换正确性")
        print("-" * 50)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / f'gpt2_{format_type}'
            out_dir.mkdir()
            
            # 执行转换
            result = converter.convert(
                input_source='hf:gpt2',
                output_format=format_type,
                output_path=str(out_dir),
                model_type='text-generation',
                validate=False  # 我们自己验证
            )
            
            if result['success']:
                print(f"转换成功: ✅")
                
                # 加载转换后的模型
                try:
                    converted_model = AutoModelForCausalLM.from_pretrained(str(out_dir))
                    converted_tokenizer = AutoTokenizer.from_pretrained(str(out_dir))
                    
                    # 获取转换后模型的权重信息
                    converted_weights = {}
                    for name, param in converted_model.named_parameters():
                        converted_weights[name] = {
                            'shape': tuple(param.shape),
                            'dtype': str(param.dtype),
                            'mean': float(param.mean().item()),
                            'std': float(param.std().item()),
                            'min': float(param.min().item()),
                            'max': float(param.max().item())
                        }
                    
                    # 权重比较分析
                    weight_analysis = compare_weights(original_weights, converted_weights)
                    
                    # 输出比较
                    with torch.no_grad():
                        converted_outputs = converted_model(**test_inputs)
                    converted_logits = converted_outputs.logits
                    
                    output_analysis = compare_outputs(original_logits, converted_logits)
                    
                    # 推理测试
                    inference_analysis = test_inference_correctness(
                        original_model, original_tokenizer,
                        converted_model, converted_tokenizer
                    )
                    
                    # 保存结果
                    results[format_type] = {
                        'success': True,
                        'weight_analysis': weight_analysis,
                        'output_analysis': output_analysis,
                        'inference_analysis': inference_analysis,
                        'model_info': {
                            'original_params': len(original_weights),
                            'converted_params': len(converted_weights),
                            'original_total_params': sum(p.numel() for p in original_model.parameters()),
                            'converted_total_params': sum(p.numel() for p in converted_model.parameters())
                        }
                    }
                    
                    print(f"权重比较: {weight_analysis['summary']}")
                    print(f"输出比较: {output_analysis['summary']}")
                    print(f"推理测试: {inference_analysis['summary']}")
                    
                except Exception as e:
                    print(f"模型加载失败: {e}")
                    results[format_type] = {
                        'success': False,
                        'error': str(e)
                    }
            else:
                print(f"转换失败: {result.get('error', '未知错误')}")
                results[format_type] = {
                    'success': False,
                    'error': result.get('error', '未知错误')
                }
    
    # 生成总结报告
    print("\n" + "=" * 60)
    print("📊 转换正确性总结报告")
    print("=" * 60)
    
    for format_type, format_name in test_formats:
        result = results[format_type]
        if result['success']:
            print(f"\n✅ {format_name}:")
            
            weight_analysis = result['weight_analysis']
            output_analysis = result['output_analysis']
            inference_analysis = result['inference_analysis']
            
            print(f"  权重正确性: {weight_analysis['correctness_score']:.1f}/10")
            print(f"  输出正确性: {output_analysis['correctness_score']:.1f}/10")
            print(f"  推理正确性: {inference_analysis['correctness_score']:.1f}/10")
            
            # 计算总体正确性评分
            overall_score = (
                weight_analysis['correctness_score'] * 0.4 +
                output_analysis['correctness_score'] * 0.4 +
                inference_analysis['correctness_score'] * 0.2
            )
            print(f"  总体正确性: {overall_score:.1f}/10")
            
            # 详细分析
            if weight_analysis['shape_mismatches']:
                print(f"    形状不匹配: {len(weight_analysis['shape_mismatches'])} 个参数")
            if weight_analysis['dtype_changes']:
                print(f"    数据类型变化: {len(weight_analysis['dtype_changes'])} 个参数")
            
            print(f"    输出差异: {output_analysis['mean_diff']:.6f}")
            print(f"    推理一致性: {inference_analysis['consistency_score']:.1%}")
            
        else:
            print(f"\n❌ {format_name}: 转换失败")
            print(f"  错误: {result['error']}")
    
    return results


def compare_weights(original_weights: Dict, converted_weights: Dict) -> Dict[str, Any]:
    """比较原始模型和转换后模型的权重"""
    
    shape_mismatches = []
    dtype_changes = []
    value_differences = []
    
    # 检查所有参数
    for name in original_weights:
        if name not in converted_weights:
            shape_mismatches.append(name)
            continue
            
        orig = original_weights[name]
        conv = converted_weights[name]
        
        # 检查形状
        if orig['shape'] != conv['shape']:
            shape_mismatches.append(name)
        
        # 检查数据类型
        if orig['dtype'] != conv['dtype']:
            dtype_changes.append(name)
        
        # 检查数值差异（对于相同形状的参数）
        if orig['shape'] == conv['shape']:
            mean_diff = abs(orig['mean'] - conv['mean'])
            std_diff = abs(orig['std'] - conv['std'])
            value_differences.append({
                'name': name,
                'mean_diff': mean_diff,
                'std_diff': std_diff
            })
    
    # 计算正确性评分
    total_params = len(original_weights)
    shape_correct = total_params - len(shape_mismatches)
    dtype_correct = total_params - len(dtype_changes)
    
    # 计算数值差异的平均值
    if value_differences:
        avg_mean_diff = sum(d['mean_diff'] for d in value_differences) / len(value_differences)
        avg_std_diff = sum(d['std_diff'] for d in value_differences) / len(value_differences)
    else:
        avg_mean_diff = 0
        avg_std_diff = 0
    
    # 评分计算
    shape_score = (shape_correct / total_params) * 4  # 40%权重
    dtype_score = (dtype_correct / total_params) * 2   # 20%权重
    value_score = max(0, 4 - avg_mean_diff * 100)      # 40%权重，基于数值差异
    
    correctness_score = min(10, shape_score + dtype_score + value_score)
    
    return {
        'shape_mismatches': shape_mismatches,
        'dtype_changes': dtype_changes,
        'value_differences': value_differences,
        'avg_mean_diff': avg_mean_diff,
        'avg_std_diff': avg_std_diff,
        'correctness_score': correctness_score,
        'summary': f"形状匹配: {shape_correct}/{total_params}, 数值差异: {avg_mean_diff:.6f}"
    }


def compare_outputs(original_logits: torch.Tensor, converted_logits: torch.Tensor) -> Dict[str, Any]:
    """比较原始模型和转换后模型的输出"""
    
    # 确保形状相同
    if original_logits.shape != converted_logits.shape:
        return {
            'correctness_score': 0,
            'mean_diff': float('inf'),
            'max_diff': float('inf'),
            'summary': f"形状不匹配: {original_logits.shape} vs {converted_logits.shape}"
        }
    
    # 计算差异
    diff = torch.abs(original_logits - converted_logits)
    mean_diff = float(diff.mean().item())
    max_diff = float(diff.max().item())
    
    # 计算正确性评分
    # 基于差异大小评分，差异越小分数越高
    if mean_diff < 1e-6:
        correctness_score = 10.0
    elif mean_diff < 1e-5:
        correctness_score = 9.0
    elif mean_diff < 1e-4:
        correctness_score = 8.0
    elif mean_diff < 1e-3:
        correctness_score = 7.0
    elif mean_diff < 1e-2:
        correctness_score = 6.0
    elif mean_diff < 1e-1:
        correctness_score = 5.0
    else:
        correctness_score = max(0, 5 - mean_diff)
    
    return {
        'mean_diff': mean_diff,
        'max_diff': max_diff,
        'correctness_score': correctness_score,
        'summary': f"平均差异: {mean_diff:.6f}, 最大差异: {max_diff:.6f}"
    }


def test_inference_correctness(
    original_model, original_tokenizer,
    converted_model, converted_tokenizer
) -> Dict[str, Any]:
    """测试推理正确性"""
    
    test_cases = [
        "Hello world",
        "The quick brown fox",
        "Machine learning is",
        "In the beginning",
        "To be or not to be"
    ]
    
    consistent_count = 0
    total_tests = len(test_cases)
    
    for test_text in test_cases:
        try:
            # 原始模型推理
            orig_inputs = original_tokenizer(test_text, return_tensors='pt')
            with torch.no_grad():
                orig_outputs = original_model(**orig_inputs)
            orig_logits = orig_outputs.logits
            
            # 转换后模型推理
            conv_inputs = converted_tokenizer(test_text, return_tensors='pt')
            with torch.no_grad():
                conv_outputs = converted_model(**conv_inputs)
            conv_logits = conv_outputs.logits
            
            # 比较输出
            if orig_logits.shape == conv_logits.shape:
                diff = torch.abs(orig_logits - conv_logits).mean().item()
                if diff < 1e-3:  # 允许一定的数值误差
                    consistent_count += 1
                    
        except Exception as e:
            print(f"推理测试失败 ({test_text}): {e}")
    
    consistency_score = consistent_count / total_tests
    correctness_score = consistency_score * 10
    
    return {
        'consistent_count': consistent_count,
        'total_tests': total_tests,
        'consistency_score': consistency_score,
        'correctness_score': correctness_score,
        'summary': f"一致性: {consistent_count}/{total_tests} ({consistency_score:.1%})"
    }


if __name__ == "__main__":
    results = test_conversion_correctness()
    
    # 保存详细结果
    with open('conversion_correctness_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 详细结果已保存到: conversion_correctness_results.json") 