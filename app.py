import streamlit as st
import av
import cv2
import numpy as np
import os
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from inspection.visualization import get_contour_orientation
from inspection import processing, metrics, config_handler, image_acquisition, video_stream

st.set_page_config(layout="wide", page_title="Part Inspection System")

ROI_SIZE = 200
ROI_SCALE = 1

detection_methods = [
    "canny",
    "watershed",
    "otsu",
    "template_match"
]
config = config_handler.load_config("config.json")
method = config.get("detection_method", "canny")
if method not in detection_methods:
    method = "canny"

reference_contour = np.load("reference_contour.npy", allow_pickle=True) if os.path.exists("reference_contour.npy") else None
reference_moments = np.load("reference_moments.npy", allow_pickle=True) if os.path.exists("reference_moments.npy") else None
reference_fd = np.load("reference_fd.npy", allow_pickle=True) if os.path.exists("reference_fd.npy") else None
reference_roi = np.load("reference_roi.npy", allow_pickle=True) if os.path.exists("reference_roi.npy") else None
reference_orientation = np.load("reference_orientation.npy") if os.path.exists("reference_orientation.npy") else None
reference_direction = np.load("reference_direction.npy", allow_pickle=True) if os.path.exists("reference_direction.npy") else np.array([1.0, 0.0])

selected_metrics = config.get("selected_metrics", ["shape_score", "rotinv_moment_dist", "fourier_dist"])
thresholds = config.get("thresholds", {m: {"ok": 0.2, "nok": 0.5} for m in selected_metrics})

metric_labels = {
    "shape_score": "Shape Match (cv2.matchShapes)",
    "rotinv_moment_dist": "Rotation-Aligned Central Moment Distance",
    "fourier_dist": "Rotation-Aligned Fourier Descriptor Distance",
    "sift_score": "SIFT Similarity Score"
}

if "live_template_img" not in st.session_state:
    st.session_state["live_template_img"] = None

tab1, tab2 = st.tabs(["Parameter Tuning & Method Comparison", "Live Inspection"])

# -------------------- TAB 1 --------------------
with tab1:
    st.header("Testing and Method Comparison")
    uploaded_file = st.file_uploader("Upload an image for testing", type=["jpg", "jpeg", "png"])
    method_tab1 = st.selectbox("Detection Method", detection_methods, index=detection_methods.index(method))
    config_metrics = config.get("selected_metrics", ["shape_score", "rotinv_moment_dist", "fourier_dist"])
    shape_metrics = ["shape_score", "rotinv_moment_dist", "fourier_dist", "sift_score"]

    selected_metrics_tab1 = st.multiselect(
        "Metrics used for decision (select one or more):",
        [f"{m}: {metric_labels[m]}" for m in shape_metrics],
        default=[f"{m}: {metric_labels[m]}" for m in config_metrics]
    )
    selected_metrics_tab1 = [m.split(":")[0] for m in selected_metrics_tab1]

    thresholds_tab1 = {}
    config_thresholds = config.get("thresholds", {})
    for m in shape_metrics:
        st.markdown(f"**{metric_labels[m]}** (lower is better, 0 = perfect match, 1 = very different)")
        ok_default = config_thresholds.get(m, {}).get("ok", 0.2 if m != "sift_score" else 0.2)
        nok_default = config_thresholds.get(m, {}).get("nok", 0.5 if m != "sift_score" else 0.5)

        ok = st.slider(f"OK threshold (max)", 0.0, 1.0, ok_default, 0.01, key=f"{m}_ok")
        nok = st.slider(f"NOK threshold (min)", 0.0, 1.0, nok_default, 0.01, key=f"{m}_nok")
        thresholds_tab1[m] = {"ok": ok, "nok": nok}
    st.markdown("#### 2. Check active metric selection and thresholds")
    st.json({m: thresholds_tab1[m] for m in selected_metrics_tab1})

    template = None
    if method_tab1 == "template_match":
        st.info("Upload a template image for matching.")
        template_file = st.file_uploader("Upload template image", type=["jpg", "jpeg", "png"], key="template")
        if template_file is not None:
            template = image_acquisition.load_image_from_file(template_file)

    contour = None
    roi = None
    moment_result = None
    overlay_up = None

    if uploaded_file is not None:
        img = image_acquisition.load_image_from_file(uploaded_file)
        roi, roi_coords = processing.crop_center(img, ROI_SIZE, ROI_SIZE)
        if method_tab1 == "template_match" and template is None:
            st.error("Please upload a template image for template matching.")
            st.stop()
        contour, mask, extra = processing.detect_contour(roi, method_tab1, template)
        moment_result = metrics.compute_metrics(
            contour, roi,
            reference_contour=reference_contour,
            reference_moments=reference_moments,
            reference_fd=reference_fd,
            reference_roi=reference_roi
        )
        angle, direction = get_contour_orientation(contour)
        ref_angle, ref_direction = get_contour_orientation(reference_contour) if reference_contour is not None else (0.0, np.array([1.0, 0.0]))
        mirrored_vertical = False
        if reference_direction is not None and direction is not None:
            mirrored_vertical = processing.is_vertically_mirrored(contour, reference_contour) #np.dot(direction, reference_direction) < 0
        metric_status = metrics.classify_alignment(moment_result, selected_metrics_tab1, thresholds_tab1)
        # Mark as NOK if mirrored
        if mirrored_vertical:
            metric_status = "NOK"

        overlay_up = processing.plot_roi_with_contour(
            roi, contour, metric_status,
            metrics_dict={m: moment_result.get(m) for m in selected_metrics_tab1},
            thresholds=thresholds_tab1,
            ref_direction=reference_direction,
            show_mirror_check=True
        )
        if mirrored_vertical:
            h = overlay_up.shape[0]
            # Draw annotation at the bottom
            cv2.putText(
                overlay_up, "VERTICAL MIRROR - NOK", (30, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4, cv2.LINE_AA
            )

        st.image(overlay_up, channels="BGR", use_column_width=True)
        st.markdown("### Metric Status")
        for m in shape_metrics:
            value = moment_result.get(m)
            ok = thresholds_tab1[m]["ok"]
            nok = thresholds_tab1[m]["nok"]
            if value is None:
                st.markdown(
                    f"<span style='color:gray'>{metric_labels[m]} ({m}): Reference missing, contour not found, or not enough data</span>",
                    unsafe_allow_html=True
                )
            else:
                color = "green" if value <= ok else "red" if value > nok else "orange"
                m_status = "OK" if value <= ok else "NOK" if value > nok else "Suspicious"
                st.markdown(
                    f"<span style='color:{color}'>{metric_labels[m]} ({m}): {value} — {m_status}</span>",
                    unsafe_allow_html=True
                )

    if st.button("Save current contour as reference"):
        if contour is None or roi is None:
            st.error("No contour or ROI to save. Please upload and process an image first.")
        else:
            ref_angle, ref_direction = get_contour_orientation(contour)
            # Force reference arrow to left (negative x-axis)
            if ref_direction[0] > 0:
                contour = np.flip(contour, axis=1)
                ref_angle, ref_direction = get_contour_orientation(contour)
            np.save('reference_contour.npy', contour)
            np.save('reference_moments.npy', metrics.rotation_invariant_moments(contour))
            np.save('reference_fd.npy', metrics.compute_fourier_descriptor(contour))
            np.save('reference_roi.npy', roi)
            np.save('reference_orientation.npy', np.array([ref_angle]))
            np.save('reference_direction.npy', ref_direction)
            st.success("Reference contour, moments, Fourier, orientation and direction saved!")

            overlay_up = processing.plot_roi_with_contour(
                roi, contour, status="OK",
                metrics_dict=None, thresholds=None, ref_direction=None, show_mirror_check=True
            )
            st.image(overlay_up, caption="Reference contour & orientation", channels="BGR")
            st.rerun()

    if st.button("Save as preferred method and thresholds"):
        config_handler.save_config("config.json", {
            "detection_method": method_tab1,
            "thresholds": thresholds_tab1,
            "selected_metrics": selected_metrics_tab1
        })
        st.success(f"Config saved with updated thresholds and selected metrics.")
        st.experimental_rerun()

# -------------------- TAB 2 --------------------
with tab2:
    st.header("Live Inspection (Smooth Video + Capture)")

    if method == "template_match":
        st.info("Upload a template image for live matching.")
        live_template_file = st.file_uploader("Live Template Image (Tab 2)", type=["jpg", "jpeg", "png"], key="live_template")
        if live_template_file is not None:
            st.session_state["live_template_img"] = image_acquisition.load_image_from_file(live_template_file)
        else:
            st.session_state["live_template_img"] = None
    else:
        st.session_state["live_template_img"] = None

    class InspectionProcessor(VideoProcessorBase):
        def __init__(self):
            self.latest_overlay = None
            self.latest_status = None
            self.latest_metrics = None
            self.template = None
            self.raw_roi = None
            self.metric_history = []
            self.n_average = 20

        def set_template(self, template_img):
            self.template = template_img

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            roi, _ = processing.crop_center(img, ROI_SIZE, ROI_SIZE)
            self.raw_roi = roi.copy()
            template_arg = self.template if method == "template_match" else None
            contour, mask, extra = processing.detect_contour(roi, method, template_arg)
            moment_result = metrics.compute_metrics(
                contour, roi,
                reference_contour=reference_contour,
                reference_moments=reference_moments,
                reference_fd=reference_fd,
                reference_roi=reference_roi
            )
            angle, direction = get_contour_orientation(contour)
            ref_angle, ref_direction = get_contour_orientation(reference_contour) if reference_contour is not None else (0.0, np.array([1.0, 0.0]))
            mirrored = False
            if reference_direction is not None and direction is not None:
                mirrored = np.dot(direction, reference_direction) < 0

            metrics_dict = {m: moment_result.get(m) for m in selected_metrics}
            self.metric_history.append(metrics_dict)
            if len(self.metric_history) > self.n_average:
                self.metric_history.pop(0)
            avg_metrics = {}
            for m in selected_metrics:
                vals = [d[m] for d in self.metric_history if d[m] is not None]
                avg_metrics[m] = float(np.mean(vals)) if vals else None

            metric_status = metrics.classify_alignment(avg_metrics, selected_metrics, thresholds)
            if mirrored:
                metric_status = "NOK"

            overlay_up = processing.plot_roi_with_contour(
                roi, contour, metric_status,
                metrics_dict=avg_metrics,
                thresholds=thresholds,
                ref_direction=reference_direction,
                show_mirror_check=True
            )
            if mirrored_vertical:
                h = overlay_up.shape[0]
                cv2.putText(
                    overlay_up, "VERTICAL MIRROR - NOK", (30, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4, cv2.LINE_AA
                )
            self.latest_overlay = overlay_up.copy()
            self.latest_status = metric_status
            self.latest_metrics = avg_metrics
            return av.VideoFrame.from_ndarray(overlay_up, format="bgr24")

    def processor_factory():
        proc = InspectionProcessor()
        if method == "template_match":
            proc.set_template(st.session_state.get("live_template_img"))
        return proc

    stream_disabled = method == "template_match" and st.session_state.get("live_template_img") is None
    if stream_disabled:
        st.warning("Please upload a template image before starting live inspection.")
    else:
        webrtc_ctx = webrtc_streamer(
            key="inspection",
            video_processor_factory=processor_factory,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                    "frameRate": {"ideal": 5, "max": 5}
                },
                "audio": False
            },
            async_processing=True,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
        )

        raw_capture_btn = st.button("Capture Frame")
        if raw_capture_btn:
            raw_frame_to_save = None
            if hasattr(webrtc_ctx, "video_processor") and webrtc_ctx.video_processor:
                raw_frame_to_save = webrtc_ctx.video_processor.raw_roi
            if raw_frame_to_save is not None:
                pics_dir = "pics"
                os.makedirs(pics_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(pics_dir, f"capture_{timestamp}.png")
                cv2.imwrite(filename, raw_frame_to_save)
                st.success(f"Frame saved: {filename}")
            else:
                st.error("No frame available yet. Wait for video to start.")
