from PIL import Image, ImageDraw, ImageFont
import os

def create_eml_icon():
    # 创建不同尺寸的图标
    sizes = [256]
    
    # 创建多个尺寸的图标
    icons = []
    
    for size in sizes:
        # 创建透明背景的图像
        img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        # 绘制圆角矩形背景
        bg_color = (0, 120, 215, 255)  # 蓝色背景
        margin = size // 10
        draw.rounded_rectangle(
            [margin, margin, size - margin, size - margin],
            radius=size//6,
            fill=bg_color
        )
        
        # 尝试加载字体
        font_size = size // 4
        try:
            # 尝试使用系统字体
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                # 尝试使用其他常见字体
                font = ImageFont.truetype("arialbd.ttf", font_size)
            except:
                try:
                    # 尝试使用微软雅黑（如果有）
                    font = ImageFont.truetype("msyh.ttc", font_size)
                except:
                    # 回退到默认字体
                    font = ImageFont.load_default()
        
        # 绘制"EML"文字
        text = "EML"
        try:
            # 获取文字边界框
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 计算文字位置（居中）
            x = (size - text_width) // 2
            y = (size - text_height) // 2
            
            # 绘制白色文字
            draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
        except:
            # 如果字体有问题，简单绘制文字
            draw.text((size//4, size//4), "EML", 
                     font=font, fill=(255, 255, 255, 255))
        
        icons.append(img)
    
    # 保存为ICO文件
    icons[0].save('eml_icon.ico', format='ICO', sizes=[(s, s) for s in sizes], 
                  append_images=icons[1:])
    
    print("ICO图标已生成: eml_icon.ico")
    return True

if __name__ == "__main__":
    create_eml_icon()