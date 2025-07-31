"""
    A simple demo to load ChatTS model and use it.
"""
import os
import time
import re
import gc
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, AutoProcessor
from pathlib import Path

# 配置参数
USE_Q8_QUANTIZATION = True
MODEL_PATH = "/data1/hujiajun/checkpoints/TimeSeriesLLM/ckpt"  # 修改为你的模型路径
DATASETS_DIR = "./datasets"
OUTPUT_DIR = "./outputs"
IMAGES_FOLDER = f"{OUTPUT_DIR}/freq_spectrum_images"
OUTPUT_FILE = f"{OUTPUT_DIR}/freq_spectrum_analysis_results.txt"
FREQ_DATA_DIR = f"{DATASETS_DIR}/cut_pp_csv"
PROMPT_FOLDERS = ["5", "6", "7"]  # JSON文件所在的文件夹

def setup_environment():
    """设置环境和加载模型"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 根据需要修改GPU设备ID
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    
    print("加载模型中...")
    
    try:
        # 加载模型
        if USE_Q8_QUANTIZATION:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, 
                trust_remote_code=True, 
                device_map='cuda:0', 
                quantization_config=quantization_config, 
                torch_dtype="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, 
                trust_remote_code=True, 
                device_map='cuda:0', 
                torch_dtype='float16'
            )
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, tokenizer=tokenizer)
        
        print("模型加载完成！")
        return model, tokenizer, processor
        
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        sys.exit(1)

def get_freq_files():
    """获取频域数据文件列表"""
    if not os.path.exists(FREQ_DATA_DIR):
        print(f"错误: 频域数据目录不存在 {FREQ_DATA_DIR}")
        sys.exit(1)
        
    freq_files = sorted([f for f in os.listdir(FREQ_DATA_DIR) if f.endswith('.csv')])
    if not freq_files:
        print(f"错误: 在 {FREQ_DATA_DIR} 中未找到CSV文件")
        sys.exit(1)
        
    file_pairs = []
    for i, freq_file in enumerate(freq_files):
        file_pairs.append({
            'freq_file': freq_file,
            'freq_path': os.path.join(FREQ_DATA_DIR, freq_file)
        })
    
    print(f"找到 {len(file_pairs)} 个频域文件需要处理")
    return file_pairs

def analyze_spectrum(model, tokenizer, processor, file_pairs):
    """分析频谱数据"""
    results = []  # 存储所有分析结果
    
    for file_index in range(len(file_pairs)):
        file_pair = file_pairs[file_index]
        freq_file = file_pair['freq_file']
        freq_path = file_pair['freq_path']
        
        print(f"处理文件 {file_index+1}/{len(file_pairs)}: {freq_file}")
        
        try:
            # 清理内存
            torch.cuda.empty_cache()
            gc.collect()
            
            # 读取频域CSV文件
            df_freq = pd.read_csv(freq_path, header=None)
            freq_data = df_freq.iloc[:, 1].to_numpy()  # 频域数据（幅值）
            
            # 获取频率轴
            try:
                freq_axis = df_freq.iloc[:, 0].to_numpy()
            except:
                freq_axis = np.arange(len(freq_data))
            
            # 绘制频域图像
            plt.figure(figsize=(12, 6))
            plt.plot(freq_axis, freq_data)
            plt.title(f'Frequency Spectrum - {freq_file}')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存图像
            image_filename = os.path.join(IMAGES_FOLDER, f"{os.path.splitext(freq_file)[0]}.png")
            plt.savefig(image_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 构建提示词
            prompt = f"""## 频谱数据分析任务：振动信号特征描述

你是一位专业的振动信号分析专家。请对以下频谱数据进行深入分析，生成详细且专业的信号特征描述报告。

### 核心分析要求：
1.  **基频及谐波**：识别基频（如50Hz、转频），并追踪各倍频成分（1、2、3倍频等），定量描述其频率和幅值（频率精确到0.1Hz，幅值保留2位小数），并分析倍频间的能量关系。
2.  **频谱结构**：描述能量分布类型（窄带/宽带）及主要集中频段；判断是离散谱线还是连续谱，并识别边频带、分析高频激发情况。
3.  **特殊现象**：识别可能的共振峰（分析带宽、品质因子），调制现象（确定载波、调制频率，分析边频带），以及非线性特征（频率耦合、能量转移、次/超谐波）。


### 输出格式要求：

【信号特征描述】
请使用连贯的技术性描述，报告中必须包含：
- 主要频率成分的定量描述
- 频谱能量分布的整体特征
- 特殊频率成分的识别


### 数据概览：
- 频谱数据长度：{len(freq_data)}个频率点
- 频率范围：{freq_axis[0]:.1f} Hz 到 {freq_axis[-1]:.1f} Hz
- 最大幅值：{np.max(freq_data):.2f}
- 平均幅值：{np.mean(freq_data):.2f}

---

**输入数据：**
TS是频域数据，长度为{len(freq_data)}，是频谱分析结果：<ts><ts/>"""

            # 应用Chat模板
            formatted_prompt = f"<|im_start|>system\nYou are a professional vibration signal analysis expert.<|im_end|><|im_start|>user\n{prompt}<|im_end|><|im_start|>assistant\n"
            
            # 转换为张量
            inputs = processor(text=[formatted_prompt], timeseries=[freq_data], padding=True, return_tensors="pt")
            inputs = {k: v.to(0) for k, v in inputs.items()}
            
            # 模型生成
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000,
                do_sample=False
            )
            
            # 解码输出
            input_len = inputs['attention_mask'][0].sum().item()
            output = outputs[0][input_len:]
            analysis = tokenizer.decode(output, skip_special_tokens=True)
            
            # 保存结果
            mode = "a" if os.path.exists(OUTPUT_FILE) else "w"
            with open(OUTPUT_FILE, mode, encoding="utf-8") as f_out:
                if mode == "w":
                    f_out.write("频谱数据分析结果\n")
                    f_out.write("="*80 + "\n\n")
                
                f_out.write(f"文件 #{file_index+1}: {freq_file}\n")
                f_out.write("-"*50 + "\n")
                f_out.write("分析结果:\n")
                f_out.write(analysis)
                f_out.write("\n\n" + "="*80 + "\n\n")
            
            # 添加到结果列表
            results.append({
                'file': freq_file,
                'analysis': analysis
            })
            
            print(f"完成！结果已保存到: {OUTPUT_FILE}")
            
            # 释放内存
            del inputs, outputs, freq_data, df_freq
            torch.cuda.empty_cache()
            gc.collect()
            
            # 等待几秒
            if file_index < len(file_pairs) - 1:
                print("等待系统释放资源...")
                time.sleep(3)
                
        except Exception as e:
            error_msg = f"处理文件 {freq_file} 时出错: {str(e)}"
            print(error_msg)
            
            # 记录错误
            mode = "a" if os.path.exists(OUTPUT_FILE) else "w"
            with open(OUTPUT_FILE, mode, encoding="utf-8") as f_out:
                if mode == "w":
                    f_out.write("频谱数据分析结果\n")
                    f_out.write("="*80 + "\n\n")
                f_out.write(f"错误: {error_msg}\n\n" + "="*80 + "\n\n")
            
            # 释放内存
            torch.cuda.empty_cache()
            gc.collect()
            
            print("发生错误，等待系统恢复...")
            time.sleep(5)
            
    return results

def update_json_with_analysis(analysis_results):
    """将分析结果更新到JSON文件中"""
    print("\n开始更新JSON文件...")
    
    # 创建文件名到分析结果的映射
    analysis_map = {}
    for result in analysis_results:
        # 获取基本文件名
        base_name = os.path.splitext(result['file'])[0]
        # 移除可能的后缀
        clean_name = base_name.replace('.mat_pinpu', '')
        # 保存映射关系
        analysis_map[clean_name.lower()] = result['analysis']
    
    modified_count = 0
    
    for folder in PROMPT_FOLDERS:
        folder_path = os.path.join("./prompt", folder)
        if not os.path.exists(folder_path):
            print(f"文件夹不存在: {folder_path}")
            continue
            
        # 获取文件夹中的所有JSON文件
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        
        for json_file in json_files:
            base_name = os.path.splitext(json_file)[0]
            clean_base = base_name.replace('-语料对', '').lower()
            json_path = os.path.join(folder_path, json_file)
            
            # 寻找匹配的分析结果
            matching_key = None
            for key in analysis_map.keys():
                if key in clean_base or clean_base in key:
                    matching_key = key
                    break
            
            if matching_key:
                try:
                    # 读取JSON文件
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    # 修改JSON内容
                    if isinstance(json_data, dict) and 'signal_data' in json_data:
                        # 添加频域分析
                        json_data['signal_data']['frequency_domain'] = analysis_map[matching_key]
                        # 将time_domain置空
                        json_data['signal_data']['time_domain'] = ""
                        
                        # 保存修改后的JSON
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(json_data, f, ensure_ascii=False, indent=2)
                        
                        modified_count += 1
                        print(f"已更新 {json_path} (匹配: {matching_key})")
                    else:
                        print(f"JSON结构不符合要求: {json_path}")
                except Exception as e:
                    print(f"处理 {json_path} 时出错: {e}")
    
    print(f"\n处理完成！共更新了 {modified_count} 个JSON文件")

def update_json_with_freq_data():
    """将频域数据添加到JSON文件中"""
    print("\n开始添加频域数据到JSON文件...")
    
    # 定义路径
    freq_domain_dir = FREQ_DATA_DIR
    
    # 确保频域数据文件夹存在
    if not os.path.exists(freq_domain_dir):
        print(f"错误：频域数据文件夹不存在: {freq_domain_dir}")
        return
        
    # 获取所有频域CSV文件
    freq_files = sorted([f for f in os.listdir(freq_domain_dir) if f.endswith('.csv')])
    
    # 创建文件名到文件路径的映射
    freq_file_map = {}
    for freq_file in freq_files:
        # 获取不带后缀的文件名
        base_name = os.path.splitext(freq_file)[0]
        # 去掉.mat_pinpu后缀（如果有的话）
        clean_name = base_name.replace('.mat_pinpu', '')
        # 保存映射关系
        freq_file_map[clean_name.lower()] = os.path.join(freq_domain_dir, freq_file)
    
    print(f"找到 {len(freq_file_map)} 个频域CSV文件")
    
    # 处理计数
    total_json_files = 0
    updated_json_files = 0
    
    # 遍历JSON文件
    for folder in PROMPT_FOLDERS:
        folder_path = os.path.join("./prompt", folder)
        if not os.path.exists(folder_path):
            print(f"文件夹不存在: {folder_path}")
            continue
            
        # 获取文件夹中的所有JSON文件
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        total_json_files += len(json_files)
        
        for json_file in json_files:
            base_name = os.path.splitext(json_file)[0]
            clean_base = base_name.replace('-语料对', '').lower()
            json_path = os.path.join(folder_path, json_file)
            
            # 寻找匹配的频域文件
            matching_key = None
            for key in freq_file_map.keys():
                if key in clean_base or clean_base in key:
                    matching_key = key
                    break
            
            if matching_key:
                # 读取频域数据
                freq_path = freq_file_map[matching_key]
                try:
                    # 读取CSV文件
                    df_freq = pd.read_csv(freq_path, header=None)
                    # 获取频域数据数组
                    freq_data = df_freq.iloc[:, 1].tolist()  # 转换为列表以便JSON序列化
                    
                    # 读取JSON文件
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    # 修改JSON内容 - 添加频域数据
                    if isinstance(json_data, dict) and 'signal_data' in json_data:
                        # 添加频域数据（限制数据长度以防止文件过大）
                        max_data_length = 5000  # 限制数据点数量
                        if len(freq_data) > max_data_length:
                            print(f"截取频域数据从 {len(freq_data)} 到 {max_data_length} 个点: {json_file}")
                            freq_data = freq_data[:max_data_length]
                        
                        # 添加频域数据
                        json_data['signal_data']['frequency_domain_data'] = freq_data
                        
                        # 保存修改后的JSON
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(json_data, f, ensure_ascii=False, indent=2)
                        
                        updated_json_files += 1
                        print(f"已更新频域数据: {json_path} (匹配: {os.path.basename(freq_path)})")
                    else:
                        print(f"JSON结构不符合要求: {json_path}")
                except Exception as e:
                    print(f"处理文件时出错 {json_path} - {freq_path}: {e}")
            else:
                print(f"未找到匹配的频域数据: {json_file}")
    
    print(f"\n处理完成！共处理 {total_json_files} 个JSON文件，更新了 {updated_json_files} 个文件的频域数据")

def send_to_deepseek(prompt_folders=None, api_key=None):
    """发送JSON格式的prompt到DeepSeek API并获取回答"""
    if prompt_folders is None:
        prompt_folders = ["6", "7"]  # 默认处理6和7文件夹
        
    if api_key is None:
        api_key = "sk-126dcaa7437f471c8a08e3a5b920aab8"  # 请替换为您的实际API密钥
    
    import requests
    from glob import glob
    from pathlib import Path
    
    DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
    
    # 获取所有JSON文件路径
    json_files = []
    for i in prompt_folders:
        folder_path = os.path.join("prompt", str(i))
        if os.path.exists(folder_path):
            json_files.extend(glob(os.path.join(folder_path, "*.json")))
    
    if not json_files:
        print("未找到任何JSON文件")
        return {}
    
    results = {}
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)
            
            print(f"正在处理: {json_file}")
            
            # 将JSON格式化为合适的请求格式
            formatted_prompt = json.dumps(prompt_data, ensure_ascii=False)
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "model": "deepseek-chat",  # 根据DeepSeek提供的模型名称调整
                "messages": [
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 3000
            }
            
            try:
                response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
                response.raise_for_status()  # 如果响应状态码不是200，将引发异常
                response_json = response.json()
                
                # 从API响应中提取回答文本
                answer = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                file_name = Path(json_file).stem
                results[file_name] = {
                    "prompt": prompt_data,
                    "response": answer
                }
                print(f"已获取回答，长度: {len(answer)}")
            except requests.exceptions.RequestException as e:
                print(f"API请求出错: {e}")
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {e}")
    
    # 保存所有结果到一个文件
    if results:
        with open('deepseek_responses.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到 deepseek_responses.json")
    
    return results

def main():
    """主函数"""
    print("欢迎使用ChatTS频谱分析演示!")
    
    # 显示可用功能菜单
    print("\n请选择要执行的功能:")
    print("1. 分析频谱数据")
    print("2. 将分析结果更新到JSON文件")
    print("3. 将频域数据添加到JSON文件")
    print("4. 使用DeepSeek API处理JSON文件")
    print("5. 执行所有功能")
    print("0. 退出")
    
    choice = input("\n请输入选项(0-5): ")
    
    if choice == "0":
        print("程序已退出")
        return
    
    if choice in ["1", "5"]:
        # 分析频谱数据
        model, tokenizer, processor = setup_environment()
        file_pairs = get_freq_files()
        analysis_results = analyze_spectrum(model, tokenizer, processor, file_pairs)
        
        if choice == "5":
            # 更新JSON文件
            update_json_with_analysis(analysis_results)
            update_json_with_freq_data()
            
            # 询问是否使用DeepSeek API
            use_deepseek = input("\n是否使用DeepSeek API处理JSON文件? (y/n): ")
            if use_deepseek.lower() == 'y':
                api_key = input("请输入DeepSeek API密钥 (直接回车使用默认密钥): ")
                api_key = api_key if api_key else None
                send_to_deepseek(api_key=api_key)
    
    elif choice == "2":
        # 读取分析结果
        if not os.path.exists(OUTPUT_FILE):
            print(f"错误: 分析结果文件不存在 {OUTPUT_FILE}")
            return
            
        # 解析分析结果文件
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            analysis_content = f.read()
        
        # 使用正则表达式解析文件内容
        analysis_blocks = re.split(r'={80}', analysis_content)
        analysis_results = []
        
        # 解析每个分析块
        for block in analysis_blocks:
            if not block.strip():
                continue
            
            file_match = re.search(r'文件 #\d+: (.+?)(?:\n|$)', block)
            if file_match:
                filename = file_match.group(1).strip()
                result_match = re.search(r'分析结果:\n([\s\S]+?)(?=\n={80}|\Z)', block)
                if result_match:
                    analysis_results.append({
                        'file': filename,
                        'analysis': result_match.group(1).strip()
                    })
        
        # 更新JSON文件
        update_json_with_analysis(analysis_results)
    
    elif choice == "3":
        # 将频域数据添加到JSON文件
        update_json_with_freq_data()
    
    elif choice == "4":
        # 使用DeepSeek API处理JSON文件
        api_key = input("请输入DeepSeek API密钥 (直接回车使用默认密钥): ")
        api_key = api_key if api_key else None
        send_to_deepseek(api_key=api_key)
    
    print("\n程序执行完毕!")

if __name__ == "__main__":
    main() 