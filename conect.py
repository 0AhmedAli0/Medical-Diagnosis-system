from flask import Flask, render_template, request
# from keras.models import load_model, Model
# import cv2
# import numpy as np
# import keras.backend as K
# from wfdb import rdrecord
# from scipy.signal import resample
# import biosppy.signals.ecg as ecg
# import matplotlib.pyplot as plt
# from torch import hub
# from os import environ
# from AFR import AFR
# from AFRLayer1D import AFRLayer1D
# from keras import backend as K
# from keras.utils import img_to_array, array_to_img
# from tensorflow import (
#     GradientTape,
#     argmax,
#     reduce_mean,
#     newaxis,
#     squeeze,
#     maximum,
#     math,
# )
# import matplotlib.cm as cm

# environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

rec_app = Flask(__name__)


@rec_app.route("/")
def Home():
    return render_template("index.html")


@rec_app.route("/index.html")
def returnHome():
    return render_template("index.html")


@rec_app.route("/index.html#About")
def About():
    return render_template("index.html")


@rec_app.route("/index.html#contact")
def Contact():
    return render_template("index.html")


@rec_app.route("/Chest.html", methods=["GET", "POST"])
def Chest():
    image_path = 0
    res_path = 0
    p = 0
    if request.method == "POST":
        imagefile = request.files["imagefile"]
        image_path = "./static/img/" + imagefile.filename
        imagefile.save(image_path)
        res_path = "./static/img/1" + imagefile.filename

        # img = cv2.imread(image_path)
        # img = cv2.resize(img, (640, 640))

        # yolo = hub.load("ultralytics/yolov5", "custom", path="best.pt")
        # results = yolo(img)

        # boxes = results.xyxy[0].numpy()
        # class_indices = results.pred[0][:, -1].long().cpu().tolist()
        # class_labels = [results.names[i] for i in class_indices]
        # scores = results.pred[0][:, 4].cpu().tolist()

        # for i, box in enumerate(boxes):
        #     x1, y1, x2, y2 = box[:4].astype(int)
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #     label = f"{class_labels[i]}: {scores[i]:.2f}"

        #     # Get the size of the label text
        #     (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        #     # Draw a white rectangle behind the label text
        #     cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1 - 5), (255, 255, 255), -1)
        #     # Draw the label text in red color
        #     cv2.putText(
        #         img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        #     )

        # cv2.imwrite(res_path, img)
        # p = "you can see a bounding box, diagnosis and confidance of each finding on the above image"
    return render_template("Chest.html", path=image_path, res_img=res_path, p=p)


@rec_app.route("/covid.html", methods=["GET", "POST"])
def Covid():
    image_path = 0
    outcome = "Result"
    res_path = 0
    p = 0

    if request.method == "POST":
        imagefile = request.files["imagefile"]
        image_path = "./static/img/" + imagefile.filename
        imagefile.save(image_path)

        # img = cv2.imread(image_path)
        # img = cv2.resize(img, (224, 224))
        # img = img / 255
        # model = load_model("covid_best_model.h5", custom_objects={"AFR": AFR})
        # pred1 = model.predict(img.reshape(1, 224, 224, 3), batch_size=1, verbose=0)
        # pred = pred1.argmax()
        # covid_labels = {0: "COVID-19", 1: "Normal", 2: "Pneumonia"}
        # outcome = (
        #     covid_labels.get(pred, "Unknown")
        #     + ", "
        #     + (pred1[0][pred] * 100).astype("str")[:6]
        #     + "%"
        # )
        # if covid_labels.get(pred, "Unknown") != "Normal":
        #     gmodel = load_model("covid_best_model.h5", custom_objects={"AFR": AFR})
        #     gmodel.layers[-1].activation = None
        #     last_conv_layer_name = "afr_3"
        #     img1 = cv2.imread(image_path)
        #     img1 = cv2.resize(img1, (224, 224))
        #     res_path = "./static/img/1" + imagefile.filename
        #     gradcam(img1, gmodel, last_conv_layer_name, cam_path=res_path)
        #     p = "you can see the area on which the diagnosis was based highlighted on the above image, note that the highlighted area doesn't includde all the infected area"
    
    return render_template("covid.html", path=image_path, Result=outcome, res_img=res_path, p=p)


@rec_app.route("/SCD.html", methods=["GET", "POST"])
def skin():
    image_path = 0
    outcome = "Result"
    if request.method == "POST":
        imagefile = request.files["imagefile"]
        image_path = "./static/img/" + imagefile.filename
        imagefile.save(image_path)
        # img = cv2.imread(image_path)
        # img = cv2.resize(img, (224, 224))
        # img = img / 255
        # model = load_model("skin_best_model.h5", custom_objects={"AFR": AFR})
        # pred1 = model.predict(img.reshape(1, 224, 224, 3), batch_size=1, verbose=0)
        # pred = (pred1 > 0.5).astype(int)
        # skin_labels = {0: "Benign", 1: "Malignant"}
        # if pred1 > 0.5:
        #     outcome = (
        #         skin_labels.get(pred[0][0], "Unknown")
        #         + ", "
        #         + (pred1[0][0] * 100).astype("str")[:6]
        #         + "%"
        #     )
        # else:
        #     outcome = (
        #         skin_labels.get(pred[0][0], "Unknown")
        #         + ", "
        #         + ((1 - pred1[0][0]) * 100).astype("str")[:6]
        #         + "%"
        #     )
    return render_template("SCD.html", path=image_path, Result=outcome)


@rec_app.route("/Heartbeat.html", methods=["GET", "POST"])
def Heartbeat():
    image_path = 0
    beat_path = 0
    outcome = "Result"
    p = 0
    if request.method == "POST":
        imagefile = request.files.getlist("imagefile")
        print("1111111")
        print(imagefile)

        # image_path = "./static/img/" + imagefile[0].filename
        # imagefile[0].save(image_path)

        # img_path = "./static/img/" + imagefile[1].filename
        # imagefile[1].save(img_path)

        # beat_path = "./static/img/1" + imagefile[0].filename

        # record = rdrecord(
        #     record_name=image_path[:-4],
        #     sampfrom=0,
        #     channels=[0],
        #     physical=False,
        #     m2s=True,
        #     smooth_frames=False,
        #     ignore_skew=False,
        #     return_res=16,
        #     force_channels=True,
        #     channel_names=None,
        #     warn_empty=False,
        # )
        # ecg_signal = record.e_d_signal[0]
        # fs = record.fs
        # new_fs = 125
        # resampled_signal = resample(ecg_signal, int(len(ecg_signal) * new_fs / fs))
        # window_size = 4 * new_fs
        # window_start = 0
        # window_end = window_start + window_size
        # ecg_window = resampled_signal[window_start:window_end]

        # ecg_norm = (ecg_window - np.min(ecg_window)) / (
        #     np.max(ecg_window) - np.min(ecg_window)
        # )
        # out = ecg.hamilton_segmenter(signal=ecg_norm, sampling_rate=125)
        # rpeaks = ecg.correct_rpeaks(
        #     signal=ecg_norm, rpeaks=out["rpeaks"], sampling_rate=125, tol=0.08
        # )
        # rr_intervals = np.diff(rpeaks)
        # T = np.median(rr_intervals)

        # rpeak = rpeaks[0][0]

        # start = rpeak
        # end = int(rpeak + T * 1.2)

        # beat = ecg_norm[start:end]

        # fixed_length = 187
        # padding_length = fixed_length - len(beat)
        # padded_beat = np.concatenate((beat, np.zeros(padding_length)))

        # plt.ioff()
        # fig = plt.figure(figsize=(28, 12))
        # plt.plot(padded_beat, linewidth=3, color="red")
        # plt.axis("off")
        # plt.savefig(beat_path[:-3] + "jpg", bbox_inches="tight", pad_inches=0)
        # plt.close(fig)

        # idecies = []
        # values = []
        # plt.ioff()
        # fig = plt.figure(figsize=(28, 12))
        # plt.plot(ecg_norm[0 : 4 * fs])
        # for i in range(start, end, 1):
        #     idecies.append(i)
        #     values.append(ecg_norm[i])
        # plt.plot(idecies, values, color="red")
        # plt.axis("off")
        # plt.savefig(image_path[:-3] + "jpg", bbox_inches="tight", pad_inches=0)
        # plt.close(fig)

        # model = load_model(
        #     "ecg_data_best_model.h5", custom_objects={"AFRLayer1D": AFRLayer1D}
        # )
        # pred1 = model.predict(padded_beat.reshape(1, 187, 1), batch_size=1, verbose=0)
        # pred = pred1.argmax()

        # ecg_labels = {
        #     0: "Normal Beat",
        #     1: "Supraventricular premature beat",
        #     2: "Premature ventricular contraction",
        #     3: "Fusion of ventricular",
        #     4: "Unknown beat",
        # }

        # outcome = (
        #     ecg_labels.get(pred, "Unknown beat")
        #     + ", "
        #     + (pred1[0][pred] * 100).astype("str")[:6]
        #     + "%"
        # )

        # image_path = image_path[:-3] + "jpg"
        # beat_path = beat_path[:-3] + "jpg"
        # p = "you can see the extracted heartbeat on which the diagnosis was based highlighted in red"

    return render_template("Heartbeat.html", path=image_path, res_path=beat_path, Result=outcome, p=p)


@rec_app.route("/Brain.html", methods=["GET", "POST"])
def brain():
    image_path = 0
    outcome = "Result"
    res_path = 0
    p=0
    if request.method == "POST":
        imagefile = request.files["imagefile"]
        image_path = "./static/img/" + imagefile.filename
        imagefile.save(image_path)
        res_path = "./static/img/1" + imagefile.filename

        # img = cv2.imread(image_path)
        # img1 = cv2.imread(image_path)
        # img2 = cv2.imread(image_path)
        # model = load_model("brain_best_model.h5")
        # img = cv2.resize(img, (64, 64))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = img / 255
        # pred1 = model.predict(img.reshape(1, 64, 64, 1), batch_size=1, verbose=0)
        # pred = pred1.argmax()
        # brain_labels = {0: "glioma", 1: "meningioma", 2: "no tumor", 3: "pituitary"}
        # out = brain_labels.get(pred, "Unknown")
        # outcome = (
        #     brain_labels.get(pred, "Unknown")
        #     + ", "
        #     + (pred1[0][pred] * 100).astype("str")[:6]
        #     + "%"
        # )

        # # tumor segmentation
        # if out != "no tumor":
        #     seg_model = load_model(
        #         "brain_seg_best_model.h5",
        #         custom_objects={
        #             "iou_coef": iou_coef,
        #             "dice_coef": dice_coef,
        #             "dice_loss": dice_loss,
        #         },
        #     )
        #     img1 = cv2.resize(img1, (256, 256))
        #     img1 = img1 / 255
        #     seg = seg_model.predict(
        #         img1.reshape(1, 256, 256, 3), batch_size=1, verbose=0
        #     )
        #     seg = np.squeeze(seg)
        #     img2 = cv2.resize(img2, (256, 256))
        #     img2[seg > 0.5] = (0, 0, 255)
        #     cv2.imwrite(res_path, img2)
        #     p='you can see the tumor area highlighted in red in the above image'
        # else:
        #     res_path = "./static/image/brain-example.jpg"

    return render_template("Brain.html", path=image_path, res_img=res_path, Result=outcome,p=p)


# def dice_coef(y_true, y_pred, smooth=100):
#     y_true_flatten = K.flatten(y_true)
#     y_pred_flatten = K.flatten(y_pred)

#     intersection = K.sum(y_true_flatten * y_pred_flatten)
#     union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
#     return (2 * intersection + smooth) / (union + smooth)


# def dice_loss(y_true, y_pred, smooth=100):
#     return -dice_coef(y_true, y_pred, smooth)


# def iou_coef(y_true, y_pred, smooth=100):
#     intersection = K.sum(y_true * y_pred)
#     sum = K.sum(y_true + y_pred)
#     iou = (intersection + smooth) / (sum - intersection + smooth)
#     return iou


# def gradcam(
#     img,
#     model,
#     last_conv_layer_name,
#     alpha=0.4,
#     cam_path="images/cam.jpg",
#     pred_index=None,
# ):
#     # img = load_img(img_path, target_size=size)
#     array = img_to_array(img)
#     img_array = np.expand_dims(array, axis=0)
#     img_array = img_array / 255

#     grad_model = Model(
#         model.inputs, [model.get_layer(last_conv_layer_name).input, model.output]
#     )

#     # Then, we compute the gradient of the top predicted class for our input image
#     # with respect to the activations of the last conv layer
#     with GradientTape() as tape:
#         last_conv_layer_output, preds = grad_model(img_array)
#         if pred_index is None:
#             pred_index = argmax(preds[0])
#         class_channel = preds[:, pred_index]

#     # This is the gradient of the output neuron (top predicted or chosen)
#     # with regard to the output feature map of the last conv layer
#     grads = tape.gradient(class_channel, last_conv_layer_output)

#     # This is a vector where each entry is the mean intensity of the gradient
#     # over a specific feature map channel
#     pooled_grads = reduce_mean(grads, axis=(0, 1, 2))

#     # We multiply each channel in the feature map array
#     # by "how important this channel is" with regard to the top predicted class
#     # then sum all the channels to obtain the heatmap class activation
#     last_conv_layer_output = last_conv_layer_output[0]
#     heatmap = last_conv_layer_output @ pooled_grads[..., newaxis]
#     heatmap = squeeze(heatmap)

#     # For visualization purpose, we will also normalize the heatmap between 0 & 1
#     heatmap = maximum(heatmap, 0) / math.reduce_max(heatmap)

#     # img1 = load_img(img_path)
#     img1 = array  # img_to_array(img1)

#     # Rescale heatmap to a range 0-255
#     heatmap = np.uint8(255 * heatmap)

#     # Use jet colormap to colorize heatmap
#     jet = cm.get_cmap("jet")

#     # Use RGB values of the colormap
#     jet_colors = jet(np.arange(256))[:, :3]
#     jet_heatmap = jet_colors[heatmap]

#     # Create an image with RGB colorized heatmap
#     jet_heatmap = array_to_img(jet_heatmap)
#     jet_heatmap = jet_heatmap.resize((img1.shape[1], img1.shape[0]))
#     jet_heatmap = img_to_array(jet_heatmap)

#     # Superimpose the heatmap on original image
#     superimposed_img = jet_heatmap * alpha + img1
#     superimposed_img = array_to_img(superimposed_img)

#     # Save the superimposed image
#     superimposed_img.save(cam_path)


if __name__ == "__main__":
    rec_app.run(debug=True, port=9000)
