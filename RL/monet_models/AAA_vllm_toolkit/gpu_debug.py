import os, ray, torch

@ray.remote(num_gpus=1)
class GpuProbe:
    def info(self):
        env = os.environ.get("CUDA_VISIBLE_DEVICES")
        cnt = torch.cuda.device_count()
        torch.cuda.set_device(0)  # must succeed inside actor
        _ = torch.empty((4,4), device="cuda:0")
        name = torch.cuda.get_device_name(0)
        return {"CUDA_VISIBLE_DEVICES": env, "device_count": cnt, "name": name}

if not ray.is_initialized():
    ray.init()
actors = [GpuProbe.remote() for _ in range(2)]  # 先起2个对照
print(ray.get([a.info.remote() for a in actors]))