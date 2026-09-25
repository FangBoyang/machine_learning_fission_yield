import joblib
import os
import numpy as np

def test_scaler_joblib(file_path, name):
    print(f"\n{'='*50}")
    print(f"测试文件 (joblib): {name}")
    print(f"路径: {file_path}")
    
    if not os.path.exists(file_path):
        print("❌ 错误: 文件不存在!")
        return

    try:
        # 使用 joblib 加载（学长用的方式）
        scaler = joblib.load(file_path)
        print(f"✅ 成功加载, 类型: {type(scaler)}")
        
        # 检查是否有 inverse_transform 方法
        if hasattr(scaler, 'inverse_transform'):
            print("✅ 拥有 inverse_transform 方法")
            
            # 尝试一个简单的反归一化测试
            test_input = np.array([[0.0]])  # 归一化空间的一个点
            try:
                result = scaler.inverse_transform(test_input)
                print(f"   测试反归一化 (输入: {test_input}) -> 输出: {result}")
                if hasattr(scaler, 'mean_'):
                    print(f"   scaler.mean_: {scaler.mean_}")
                if hasattr(scaler, 'scale_'):
                    print(f"   scaler.scale_: {scaler.scale_}")
            except Exception as e:
                print(f"⚠️ 反归一化测试失败: {e}")
        else:
            print("❌ 没有 inverse_transform 方法，不是有效的 scaler")
            print(f"   实际内容: {scaler}")
            
    except Exception as e:
        print(f"❌ 加载失败: {e}")

if __name__ == "__main__":
    base_dir = r"F:\computer_science\machine_learning_fission_yield\data"
    
    files = {
        "standard_scalerZ.pkl": os.path.join(base_dir, "standard_scalerZ.pkl"),
        "standard_scalerA.pkl": os.path.join(base_dir, "standard_scalerA.pkl"),
        "standard_scalerE.pkl": os.path.join(base_dir, "standard_scalerE.pkl"),
        "yield_scaler.pkl": os.path.join(base_dir, "yield_scaler.pkl")
    }
    
    print("🔬 使用 joblib 测试 scaler 文件...")
    for name, path in files.items():
        test_scaler_joblib(path, name)
    print("\n" + "="*50)