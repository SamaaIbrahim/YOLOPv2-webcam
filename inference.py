import depthai as dai
import cv2
import numpy as np
import time


class InferenceYolopv2:
    def __init__(self, blob_path: str, nn_width: int = 256, nn_height: int = 256):
        self.nn_width = nn_width
        self.nn_height = nn_height
        self.blob_path = blob_path
        self.device = None
        self.qRgb = None
        self.qNN = None

        self.pipeline = self.create_pipeline()
        self.start_device()

    def create_pipeline(self):
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2021_4)

        # Color Camera node
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(self.nn_width, self.nn_height)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # Neural Network node
        yolopv2_lane_nn = pipeline.create(dai.node.NeuralNetwork)
        yolopv2_lane_nn.setBlobPath(self.blob_path)
        yolopv2_lane_nn.setNumInferenceThreads(2)
        yolopv2_lane_nn.input.setBlocking(False)
        yolopv2_lane_nn.setNumPoolFrames(4)

        # XLink Outputs for debugging / visualization
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        xout_nn = pipeline.create(dai.node.XLinkOut)
        xout_nn.setStreamName("nn")
        yolopv2_lane_nn.out.link(xout_nn.input)

        # Connect camera preview to NN input
        cam_rgb.preview.link(yolopv2_lane_nn.input)


        return pipeline

    def start_device(self):

        self.device = dai.Device(self.pipeline)

        self.qRgb = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        self.qNN = self.device.getOutputQueue(name="nn", maxSize=1, blocking=False)


    def run_inference(self):
        start_time = time.monotonic()
        counter, fps = 0, 0

        try:
            while True:
                # Get frames from queues
                rgb_frame = self.qRgb.get().getCvFrame()
                nn_data = self.qNN.get()

                # Get NN Output (from layer "759")
                lane_mask_data = nn_data.getLayerFp16("759")

                # Convert and reshape the lane mask
                lane_mask = np.array(lane_mask_data).reshape(1, 1, self.nn_height, self.nn_width)
                lane_mask = lane_mask.squeeze().astype(np.uint8)

                # Create a colored mask (Green)
                mask_colored = np.zeros((self.nn_height, self.nn_width, 3), dtype=np.uint8)
                mask_colored[lane_mask > 0] = [0, 255, 0]

                # Overlay the mask on the original RGB frame
                overlay_frame = rgb_frame.copy()
                overlay_frame[lane_mask > 0] = mask_colored[lane_mask > 0]

                scale_factor = 2
                resized_overlay = cv2.resize(
                    overlay_frame,
                    (self.nn_width * scale_factor, self.nn_height * scale_factor)
                )



                # Display the overlay
                cv2.imshow("YOLOPv2 Lane Mask ", resized_overlay)
                print( lane_mask)


        finally:
            self.cleanup()

    def cleanup(self):

        cv2.destroyAllWindows()
        if self.device:
            del self.device



if __name__ == "__main__":
    yolopv2 = InferenceYolopv2(blob_path="yolopv2.blob")

    yolopv2.run_inference()

