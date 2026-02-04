import cv2
from rgbd_mocap.RgbdImages import RgbdImages
import os
from helper import Projector, inpaint

def main():
    tracking_config = r"D:\Documents\Programmation\pose-estimation-finetune\annotations\data\test\label_config.json"
    rgbd = RgbdImages()
    rgbd.initialize_tracking(
        tracking_config,  # path,
        build_kinematic_model=False,
        use_kalman=True,  # al == "filtered",
        use_optical_flow=True,
        multi_processing=False,
        images_path=r"D:\Documents\Programmation\pose-estimation-finetune\annotations\data\test\images",
        from_dlc=False,
        ignore_all_checks=False,
        downsample_ratio=1,
        marker_to_exclude=["thigh", "knee", "ankle"],
    )
    projector = Projector(tracking_config)
    mapping_real_virt = [
        ['thigh', ['thigh_0', 'thigh_1']],
        ['knee', ['thigh_0', 'thigh_1']],
        ['ankle', ['leg_0', 'leg_1']]
    ]
    projector.project_real_in_local(mapping_real_virt)

    while True:
        frame, res = rgbd.get_frames(
            fit_model=False,
            show_image=False,
            save_data=False,
            save_video=False,
            file_path=rgbd.tracking_config["directory"] + os.sep + f"markers_pose.bio",
            video_name=f"video_labeled",
        )
        if not frame:
            if rgbd.video_object is not None:
                rgbd.video_object.release()
            cv2.destroyAllWindows()
            break
        marker_names = res["markers_names"]
        marker_pose = [res["markers_in_pixel"][:2, i, 0] for i in range(len(marker_names))]
        proj = projector.project_real(mapping_real_virt, marker_pose, marker_names)
        im = rgbd.process_image.frames.color
        # im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        for pos in marker_pose:
            im = cv2.circle(im, tuple(pos.astype(int)), 10, (0, 255, 0), -1)
        for p in proj:
            im = cv2.circle(im, tuple(p[:2].astype(int)), 10, (255, 0, 0), -1)
        im = inpaint(im, marker_pose, 20)
        cv2.imshow("frame", im)
        cv2.waitKey(1)
        iter = rgbd.iter
        if iter % 500 == 0:
            print("nb iterations:", iter)


if __name__ == "__main__":
    main()
