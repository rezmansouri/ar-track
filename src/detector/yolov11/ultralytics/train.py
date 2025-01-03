import sys
from ultralytics import YOLO


def main():
    image_size, batch_size, scale, devices = (
        int(sys.argv[1]),
        int(sys.argv[2]),
        sys.argv[3],
        sys.argv[4],
    )
    devices = [int(d) for d in devices]
    model = YOLO(f"./yolo11{scale}.yaml")
    model.train(
        data="hek.yaml",
        batch=batch_size,
        val=True,
        plots=True,
        pretrained=False,
        close_mosaic=0,
        epochs=100,
        time=12,
        save_period=1,
        imgsz=image_size,
        single_cls=True,
        augment=False,
        hsv_h=0,
        hsv_s=0,
        hsv_v=0,
        degrees=5,
        translate=0,
        scale=0,
        shear=0,
        perspective=0,
        flipud=0,
        fliplr=0.5,
        bgr=0,
        mosaic=0,
        mixup=0,
        copy_paste=0,
        auto_augment=False,
        erasing=0,
        crop_fraction=0,
        device=devices,
    )


if __name__ == "__main__":
    main()
