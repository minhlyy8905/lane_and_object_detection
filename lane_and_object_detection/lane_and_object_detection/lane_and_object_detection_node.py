#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
import cv2
import os
import numpy as np

class LaneAndObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_and_object_detection_node')

        # Load YOLOv8 model
        model_path = os.path.join(
            get_package_share_directory('lane_and_object_detection'),
            'best.pt'
        )
        self.get_logger().info(f"🚀 Loading YOLOv8 model from: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = list(self.model.names.values())

        self.bridge = CvBridge()

        # Sub & Pub
        self.subscriber = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(Image, 'combined/image_raw', 10)

        cv2.namedWindow("Lane & Object Detection", cv2.WINDOW_NORMAL)
        self.get_logger().info("🎯 Node LaneAndObjectDetection đã khởi động.")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            visual = frame.copy()

            # =============== Phát hiện vật thể (YOLOv8) ===============
            results = self.model.predict(frame, conf=0.4, show=False, verbose=False)
            for result in results:
                for box in result.boxes.cpu().numpy():
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    score = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.class_names[class_id]

                    color = (0, 255, 0)  # Green for object
                    cv2.rectangle(visual, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(visual, f"{class_name} {score:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # =============== Phát hiện làn đường (OpenCV) ===============
            lane_visual = self.detect_lanes(frame)
            visual = cv2.addWeighted(visual, 1, lane_visual, 1, 0)

            # Publish và hiển thị
            self.publisher.publish(self.bridge.cv2_to_imgmsg(visual, encoding='bgr8'))
            cv2.imshow("Lane & Object Detection", visual)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Lỗi xử lý ảnh: {e}")

    def detect_lanes(self, frame):
        # Resize và xử lý cơ bản
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Mask vùng ROI
        height, width = frame.shape[:2]
        mask = np.zeros_like(edges)
        polygon = np.array([
[(0, height), (width, height), (width, int(height*0.6)), (0, int(height*0.6))]
        ], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked = cv2.bitwise_and(edges, mask)

        # Dò line bằng Hough
        lines = cv2.HoughLinesP(masked, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=150)
        lane_overlay = np.zeros_like(frame)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lane_overlay, (x1, y1), (x2, y2), (255, 0, 0), 4)  # Blue

        return lane_overlay

def main(args=None):
    rclpy.init(args=args)
    node = LaneAndObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()