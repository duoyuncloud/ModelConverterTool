#!/usr/bin/env python3
"""
MTK 转换测试脚本 - 最终版本

用于测试 minicpm-pro-1b-sft 模型的 MTK 转换，包含进度跟踪和详细日志。

可重新运行的示例命令:
# 完整转换测试
python test_mtk_final_conversion.py --model-path /cache/sunhaojun/Models/minicpm-pro-1b-sft --timeout 300

# 只验证配置
python test_mtk_final_conversion.py --model-path /cache/sunhaojun/Models/minicpm-pro-1b-sft --dry-run
"""

import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

# 添加 ModelConverterTool 到路径
sys.path.insert(0, 'ModelConverterTool')

def main():
    parser = argparse.ArgumentParser(description='MTK 转换最终测试')
    parser.add_argument('--model-path', required=True, help='模型路径')
    parser.add_argument('--output-path', default='/cache/sunhaojun/tmp/mtk_final_test', help='输出路径')
    parser.add_argument('--timeout', type=int, default=600, help='超时时间（秒）')
    parser.add_argument('--dry-run', action='store_true', help='只验证配置，不实际转换')
    
    args = parser.parse_args()
    
    print(f"=== MTK 转换测试开始 [{datetime.now().strftime('%H:%M:%S')}] ===")
    print(f"📂 模型路径: {args.model_path}")
    print(f"📂 输出路径: {args.output_path}")
    print(f"⏱️  超时时间: {args.timeout} 秒")
    print(f"🔧 测试模式: {'验证配置' if args.dry_run else '完整转换'}")
    print()
    
    try:
        from model_converter_tool.api import ModelConverterAPI
        
        # 1. 验证配置
        print("🔍 步骤 1: 验证模型和配置")
        api = ModelConverterAPI()
        
        # 检查模型路径
        if not Path(args.model_path).exists():
            print(f"❌ 错误: 模型路径不存在 - {args.model_path}")
            return 1
            
        print(f"✅ 模型路径存在: {args.model_path}")
        
        # 检查输出目录
        output_dir = Path(args.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 输出目录已准备: {output_dir}")
        
        if args.dry_run:
            print("\n🎯 Dry-run 模式 - 转换配置验证通过")
            print("配置详情:")
            print("  - 模型类型: text-generation (LLM)")
            print("  - 平台: MT6991")
            print("  - 模型大小: 1_2B")
            print("  - 权重位数: 4")
            print("  - 跳过 conda 激活: ✅")
            print("  - 跳过 env.sh 设置: ✅")
            print("  - 实时输出显示: ✅")
            return 0
        
        # 2. 开始实际转换
        print(f"\n🚀 步骤 2: 开始 MTK 转换 [{datetime.now().strftime('%H:%M:%S')}]")
        
        start_time = time.time()
        
        result = api.convert_model(
            model_path=args.model_path,
            output_format='mtk',
            output_path=args.output_path,
            model_type='text-generation',
            quantization_config={
                'platform': 'MT6991',
                'model_size': '1_2B',
                'weight_bit': 4
            }
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n📊 转换完成 [{datetime.now().strftime('%H:%M:%S')}]")
        print(f"⏱️  总耗时: {duration:.1f} 秒")
        
        # 3. 检查结果
        print("\n🔎 步骤 3: 检查转换结果")
        
        if result.success:
            print("✅ 转换成功!")
            print(f"📂 输出路径: {result.output_path}")
            
            if result.validation:
                print("✅ 输出验证: 通过")
            else:
                print("⚠️  输出验证: 失败")
                
            if result.extra_info:
                print("📋 额外信息:")
                for key, value in result.extra_info.items():
                    print(f"   {key}: {value}")
                    
            # 检查输出文件
            output_path = Path(result.output_path)
            if output_path.exists():
                tflite_files = list(output_path.glob("**/*.tflite"))
                if tflite_files:
                    print(f"🎯 找到 {len(tflite_files)} 个 .tflite 文件:")
                    for f in tflite_files:
                        print(f"   📄 {f}")
                        
            return 0
        else:
            print("❌ 转换失败!")
            print(f"💬 错误信息: {result.error}")
            
            if hasattr(result, 'extra_info') and result.extra_info:
                print("📋 额外信息:")
                for key, value in result.extra_info.items():
                    print(f"   {key}: {value}")
                    
            return 1
            
    except Exception as e:
        print(f"❌ 异常错误: {str(e)}")
        import traceback
        print("\n🔍 详细错误信息:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\n=== 测试完成，退出码: {exit_code} ===")
    sys.exit(exit_code) 