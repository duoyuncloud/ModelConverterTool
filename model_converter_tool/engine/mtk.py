import os
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging

from ..converter import ConversionResult

logger = logging.getLogger(__name__)


def convert_hf_to_mtk(
    model, 
    tokenizer, 
    model_name: str, 
    output_path: str, 
    model_type: str = "auto",
    device: str = "auto",
    quantization: Optional[str] = None,
    use_large_calibration: bool = False,
    quantization_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> tuple:
    """
    将 HuggingFace 模型转换为 MTK 格式
    
    Args:
        model: HuggingFace 模型对象 (此实现中不直接使用)
        tokenizer: HuggingFace tokenizer 对象 (此实现中不直接使用)
        model_name: 模型路径
        output_path: 输出路径
        model_type: 模型类型 ("llm", "vlm", "auto")
        device: 设备 (此实现中不直接使用)
        quantization_config: 量化配置字典
        **kwargs: 其他参数
        
    Returns:
        tuple: (success: bool, extra_info: dict or str)
               - success=True时，extra_info包含转换详情
               - success=False时，extra_info是错误消息字符串
        
    示例使用方法:
        # LLM 转换 (使用默认路径)
        python -c "
        from model_converter_tool.api import ModelConverterAPI
        api = ModelConverterAPI()
        result = api.convert_model(
            model_path='/path/to/llm/model',
            output_format='mtk',
            output_path='/path/to/output',
            model_type='text-generation',
            quantization_config={
                'platform': 'MT6991',
                'model_size': '1_2B',
                'weight_bit': 4
            }
        )
        print(result)
        "
        
        # VLM 转换 (自定义 mtk_cloud 路径)
        python -c "
        from model_converter_tool.api import ModelConverterAPI
        api = ModelConverterAPI()
        result = api.convert_model(
            model_path='/path/to/vlm/model',
            output_format='mtk',
            output_path='/path/to/output',
            model_type='image-classification',
            quantization_config={
                'platform': 'MT6897',
                'model_size': '1_6B',
                'weight_bit': 4,
                'mtk_cloud_path': '/custom/path/to/mtk_cloud'
            }
        )
        print(result)
        "
        
        # 或使用环境变量
        # export MTK_CLOUD_PATH=/custom/path/to/mtk_cloud
        # python -c "转换代码..."
    """
    try:
        # 1. 获取和验证配置
        config = _get_mtk_config(quantization_config)
        
        # 2. 映射和检测模型类型
        mtk_model_type = _map_model_type_to_mtk(model_type, model_name)
        
        logger.info(f"开始 MTK 转换: {model_name} -> {output_path}")
        logger.info(f"模型类型: {model_type} -> {mtk_model_type}, 平台: {config['platform']}")
        
        # 3. 设置环境变量
        env = _setup_mtk_environment(model_name, config, mtk_model_type)
        
        # 4. 执行 mtk_cloud 转换
        mtk_cloud_path = config.get('mtk_cloud_path')
        result = _execute_mtk_conversion(mtk_model_type, env, mtk_cloud_path)
        
        if result.returncode == 0:
            # 5. 验证转换结果
            validation_result = validate_mtk_file(output_path)
            
            # 返回 (success, extra_info) 格式
            return True, {
                "model_type": mtk_model_type,
                "platform": config["platform"],
                "model_size": config["model_size"],
                "validation": validation_result
            }
        else:
            # 转换失败，返回 (success, error_message) 格式
            return False, f"MTK 转换失败: {result.stderr}"
            
    except Exception as e:
        logger.error(f"MTK 转换异常: {str(e)}")
        return False, f"MTK 转换异常: {str(e)}"


def validate_mtk_file(file_path: str) -> bool:
    """
    验证 MTK 转换结果
    
    Args:
        file_path: 输出文件路径
        
    Returns:
        bool: 验证是否通过
    """
    try:
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"输出路径不存在: {file_path}")
            return False
        
        # 检查 tflite 文件 (MTK 转换的主要输出)
        tflite_files = list(path.glob("**/*.tflite"))
        if len(tflite_files) > 0:
            logger.info(f"找到 {len(tflite_files)} 个 .tflite 文件")
            return True
        
        # 检查其他可能的输出文件
        output_files = list(path.glob("**/*"))
        if len(output_files) > 0:
            logger.info(f"找到 {len(output_files)} 个输出文件")
            return True
        
        logger.warning("未找到有效的 MTK 输出文件")
        return False
        
    except Exception as e:
        logger.error(f"验证 MTK 文件时发生错误: {str(e)}")
        return False


def _get_mtk_config(quantization_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """获取并验证 MTK 配置"""
    default_config = {
        "platform": "MT6991",
        "model_size": "1_2B", 
        "weight_bit": 4,
        "clen": 512,
        "plen": 128,
        "GPTQ_MODELPATH": "/cache/sunhaojun/ModelConverterMTK/1_2B_test",
        "cloud_weight_num": 1024,
        "cloud_act_num": 256,
        "mtk_batch_size": 2,
        "OUTPUT_DIR": "/cache/sunhaojun/output",
        "disable_pgptq": False,
        "mtk_cloud_path": None  # 新增: 自定义 mtk_cloud 路径
    }
    
    if quantization_config:
        default_config.update(quantization_config)
    
    # 验证必要参数
    valid_platforms = ["MT6991", "MT6989", "MT6897"]
    if default_config["platform"] not in valid_platforms:
        raise ValueError(f"不支持的平台: {default_config['platform']}, 支持的平台: {valid_platforms}")
    
    valid_sizes = ["0_5B", "0_9B", "1_2B", "1_6B", "8B", "0_58B"]
    if default_config["model_size"] not in valid_sizes:
        logger.warning(f"模型大小 {default_config['model_size']} 可能不被支持，支持的大小: {valid_sizes}")
    
    return default_config


def _map_model_type_to_mtk(model_type: str, model_path: str) -> str:
    """
    将标准模型类型映射到 MTK 模型类型
    
    Args:
        model_type: 标准模型类型
        model_path: 模型路径
        
    Returns:  
        str: MTK 模型类型 ("llm" 或 "vlm")
    """
    if model_type == "text-generation":
        return "llm"
    elif model_type == "image-classification":
        return "vlm"
    elif model_type == "auto":
        return _detect_mtk_model_type(model_path)
    else:
        # 对于其他类型，尝试自动检测
        logger.warning(f"未知模型类型 {model_type}，尝试自动检测")
        return _detect_mtk_model_type(model_path)


def _detect_mtk_model_type(model_path: str) -> str:
    """
    检测 MTK 模型类型 (LLM vs VLM)
    
    Args:
        model_path: 模型路径
        
    Returns:
        str: "llm" 或 "vlm"
    """
    try:
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 检查是否包含视觉相关配置
            vision_keys = ["vision_config", "image_size", "image_processor", "vision_tower"]
            if any(key in config for key in vision_keys):
                logger.info("检测到视觉相关配置，判断为 VLM 模型")
                return "vlm"
            
            # 检查模型架构名称
            model_type = config.get("model_type", "").lower()
            if "vision" in model_type or "vlm" in model_type or "multimodal" in model_type:
                logger.info(f"根据 model_type ({model_type}) 判断为 VLM 模型")
                return "vlm"
        
        # 根据模型路径名称推断
        path_str = str(model_path).lower()
        if any(keyword in path_str for keyword in ["vision", "vlm", "multimodal", "3o"]):
            logger.info("根据路径名称判断为 VLM 模型")
            return "vlm"
        
        logger.info("默认判断为 LLM 模型")
        return "llm"
        
    except Exception as e:
        logger.warning(f"检测模型类型时发生错误: {str(e)}, 默认为 LLM")
        return "llm"


def _detect_model_size(model_path: str) -> str:
    """检测模型大小"""
    path_str = str(model_path).lower()
    
    size_mappings = {
        "0.5b": "0_5B",
        "0_5b": "0_5B", 
        "0.58b": "0_58B",
        "0_58b": "0_58B",
        "0.9b": "0_9B",
        "0_9b": "0_9B",
        "1.2b": "1_2B",
        "1_2b": "1_2B",
        "1.6b": "1_6B", 
        "1_6b": "1_6B",
        "8b": "8B"
    }
    
    for key, value in size_mappings.items():
        if key in path_str:
            logger.info(f"根据路径检测到模型大小: {value}")
            return value
    
    logger.info("未能检测到模型大小，默认使用 1_2B")
    return "1_2B"


def _get_mtk_cloud_path(custom_path: Optional[str] = None) -> Path:
    """
    获取 mtk_cloud 目录路径，支持多种方式指定
    
    Args:
        custom_path: 自定义路径 (来自参数配置)
        
    Returns:
        Path: mtk_cloud 目录路径
        
    优先级:
        1. 参数传入的路径
        2. 环境变量 MTK_CLOUD_PATH
        3. 默认相对路径计算
    """
    # 方案1: 优先使用参数传入的路径
    if custom_path:
        mtk_path = Path(custom_path)
        if mtk_path.exists():
            logger.info(f"使用参数指定的 mtk_cloud 路径: {mtk_path}")
            return mtk_path
        else:
            logger.warning(f"参数指定的路径不存在: {mtk_path}")
    
    # 方案2: 使用环境变量
    if 'MTK_CLOUD_PATH' in os.environ:
        mtk_path = Path(os.environ['MTK_CLOUD_PATH'])
        if mtk_path.exists():
            logger.info(f"使用环境变量指定的 mtk_cloud 路径: {mtk_path}")
            return mtk_path
        else:
            logger.warning(f"环境变量指定的路径不存在: {mtk_path}")
    
    # 方案3: 默认相对路径计算 (向后兼容)
    current_file = Path(__file__)
    default_path = current_file.parent.parent.parent.parent / 'mtk_cloud'
    logger.info(f"使用默认相对路径计算的 mtk_cloud 路径: {default_path}")
    return default_path


def _setup_mtk_environment(model_path: str, config: Dict[str, Any], model_type: str) -> Dict[str, str]:
    """设置 MTK 转换环境变量"""
    env = os.environ.copy()
    
    # 基础参数
    env.update({
        'MODEL_PATH': model_path,
        'PLATFORM': config['platform'],
        'MODEL_KIND': 'word' if model_type == 'llm' else 'multimodal',
        'MODEL_SIZE': config['model_size'],
        'WEIGHT_BIT': str(config['weight_bit']),
        'CLEN': str(config['clen']),
        'PLEN': str(config['plen']),
        'CLOUD_WEIGHT_NUM': str(config['cloud_weight_num']),
        'CLOUD_ACT_NUM': str(config['cloud_act_num']),
        'MTK_BATCH_SIZE': str(config['mtk_batch_size']),
        'OUTPUT_DIR': config['OUTPUT_DIR'],
        'DISABLE_PGPTQ': 'true' if config['disable_pgptq'] else 'false',
        'IS_LLAMA': 'false',  # 根据需要调整
        'RATE': '0',
        'SKIP_ENV_SETUP': 'true',  # 跳过 env.sh 环境设置步骤
        'USE_OPTIMIZED_CONVERT': 'true',  # 使用优化版本的转换脚本
        'GPTQ_MODELPATH': config['GPTQ_MODELPATH']  # 添加 GPTQ 模型路径
    })
    
    # 设置 conda 环境 - 在 ModelConverterTool 中跳过 conda 激活步骤
    if 'CONDA_ENV' not in env:
        env['CONDA_ENV'] = 'SKIP_CONDA_ACTIVATION'
        logger.info("设置跳过 conda 激活步骤，使用当前 Python 环境")
    
    logger.info("MTK 环境变量设置完成")
    return env


def _execute_mtk_conversion(model_type: str, env: Dict[str, str], mtk_cloud_path: Optional[str] = None) -> subprocess.CompletedProcess:
    """执行 MTK 转换并实时显示输出"""
    # 确定 mtk_cloud 路径
    mtk_cloud_path = _get_mtk_cloud_path(mtk_cloud_path)
    
    if not mtk_cloud_path.exists():
        raise FileNotFoundError(f"未找到 mtk_cloud 目录: {mtk_cloud_path}")
    
    # 选择对应的转换脚本
    if model_type == 'llm':
        script_name = 'run_example_llm_cybertron.sh'
    elif model_type == 'vlm': 
        script_name = 'run_example_vlm_cybertron.sh'
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    script_path = mtk_cloud_path / script_name
    
    if not script_path.exists():
        raise FileNotFoundError(f"未找到转换脚本: {script_path}")
    
    logger.info(f"执行 MTK 转换脚本: {script_path}")
    print(f"🔧 开始执行 MTK 转换脚本: {script_name}")
    
    # 使用 Popen 实时显示输出
    import sys
    process = subprocess.Popen(
        ['bash', str(script_path)],
        env=env,
        cwd=str(mtk_cloud_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # 实时读取和显示输出
    stdout_lines = []
    print("📋 转换过程输出:")
    print("-" * 60)
    
    try:
        for line in iter(process.stdout.readline, ''):
            line = line.rstrip()
            if line:
                print(f"  {line}")
                stdout_lines.append(line)
                sys.stdout.flush()  # 强制刷新输出
        
        # 等待进程完成
        return_code = process.wait()
        
    except KeyboardInterrupt:
        print("\n⚠️  转换被用户中断")
        process.terminate()
        return_code = -1
    
    print("-" * 60)
    print(f"🏁 转换脚本执行完成，返回码: {return_code}")
    
    # 构造一个类似 subprocess.run 返回的结果对象
    class FakeCompletedProcess:
        def __init__(self, returncode, stdout, stderr=""):
            self.returncode = returncode
            self.stdout = '\n'.join(stdout) if isinstance(stdout, list) else stdout
            self.stderr = stderr
    
    result = FakeCompletedProcess(return_code, stdout_lines)
    
    logger.info(f"MTK 转换完成，返回码: {result.returncode}")
    if result.stdout:
        logger.info(f"标准输出长度: {len(result.stdout)} 字符")
    
    return result
