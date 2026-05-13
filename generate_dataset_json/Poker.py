import os
import json

class PokerSolver(object):
    # 这里定义你的类别名称，必须与文件夹名一致
    CLSNAMES = ['card']

    def __init__(self, root='D:\\test\\AnomalyCLIP\\PokerData'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            
            # 只处理 train 文件夹
            for phase in ['train']:
                cls_info = []
                phase_dir = f'{cls_dir}/{phase}'
                
                if not os.path.exists(phase_dir):
                    continue

                # 遍历 good, rotten 等子文件夹
                species = os.listdir(phase_dir)
                for specie in species:
                    is_abnormal = True if specie not in ['good'] else False
                    
                    img_dir = f'{phase_dir}/{specie}'
                    if not os.path.isdir(img_dir): continue
                    
                    # 获取所有图片文件 (支持 jpg, png, jpeg)
                    img_names = [x for x in os.listdir(img_dir) if x.lower().endswith(('.jpg', '.png', '.jpeg'))]
                    img_names.sort()
                    
                    for img_name in img_names:
                        # 1. 构造图片相对路径
                        # 例如: orange/train/rotten/001.jpg
                        img_path = f'{cls_name}/{phase}/{specie}/{img_name}'
                        
                        mask_path = ''
                        if is_abnormal:
                            # 2. 关键修改：精准推断掩码文件名
                            # 逻辑：把图片扩展名换成 .png，并去 ground_truth 文件夹找
                            # 原图: 001.jpg -> 掩码: 001.png
                            mask_name = os.path.splitext(img_name)[0] + "_mask"+ ".png"
                            
                            # 构造掩码绝对路径用于检查是否存在
                            gt_abs_path = os.path.join(self.root, cls_name, 'ground_truth', specie, mask_name)
                            
                            if os.path.exists(gt_abs_path):
                                # 构造掩码相对路径写入 json
                                mask_path = f'{cls_name}/ground_truth/{specie}/{mask_name}'
                            else:
                                # 如果是异常样本但没找到掩码，打印警告（也可以选择报错）
                                print(f"Warning: 找不到掩码 -> {gt_abs_path}")
                                mask_path = '' 

                        info_img = dict(
                            img_path=img_path,
                            mask_path=mask_path,
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                
                info[phase][cls_name] = cls_info
        
        # 写入文件
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
            print(f"成功生成: {self.meta_path}")

if __name__ == '__main__':
    # 记得改成你自己的实际路径
    runner = PokerSolver(root='D:\\test\\AnomalyCLIP\\PokerData')
    runner.run()