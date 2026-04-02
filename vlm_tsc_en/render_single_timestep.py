'''
Author: WANG Maonan
Date: 2025-07-29 17:53:18
LastEditors: WANG Maonan
Description: 单步渲染
LastEditTime: 2025-07-30 12:27:03
'''
import os
import sys
import bpy
import time
import argparse


tshub_path = tshub_path = "C:/Users/Utente/Documents/VLMLight_Project/VLMLight/"
sys.path.insert(0, tshub_path + "tshub/tshub_env3d/")

from vis3d_blender_render import TimestepRenderer, VehicleManager
MODELS_BASE_PATH = f"{tshub_path}/tshub/tshub_env3d/_assets_3d/vehicles_high_poly/" # 需要渲染的模型 (high poly or low poly), 渲染速度不同

# 添加命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='高精度时间步渲染器')

    parser.add_argument('--timestep_path', type=str, required=True, help='时间步数据文件夹路径')
    parser.add_argument('--resolution', type=int, default=480, help='渲染分辨率')
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:        
        # 初始化
        vehicle_mgr = VehicleManager(MODELS_BASE_PATH)
        renderer = TimestepRenderer(
            resolution=args.resolution,
            render_mask=False, 
            render_depth=False
        )
        
        # 检查路径
        if not os.path.exists(args.timestep_path):
            print(f"⚠️ 文件夹不存在: {args.timestep_path}")
            return
        
        json_path = os.path.join(args.timestep_path, "data.json")
        if not os.path.exists(json_path):
            print(f"⚠️ JSON文件不存在: {json_path}")
            return
        
        # 渲染
        print(f"\n🕒 开始渲染: {args.timestep_path}")
        start_time = time.time()
        
        vehicles = vehicle_mgr.load_vehicles(json_path)
        if not vehicles:
            print(f"⚠️ 未加载车辆: {args.timestep_path}")
        
        renderer.render_timestep(args.timestep_path)
        
        elapsed = time.time() - start_time
        print(f"✅ 渲染完成: {args.timestep_path} (耗时: {elapsed:.2f}秒)")
        
    except Exception as e:
        print(f"🔥 渲染错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if '--' in sys.argv:
        # 提取'--'之后的参数
        args = sys.argv[sys.argv.index('--') + 1:]
        sys.argv = [sys.argv[0]] + args
    
    main()