# Load the pre-trained model and optimizer
model = smp.UnetPlusPlus(
    encoder_name="resnet34",        
    encoder_weights="imagenet",     
    in_channels=3,                  
    classes=3     
)

optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

checkpoint = torch.load('/kaggle/input/unet-model/model.pth')
optimizer.load_state_dict(checkpoint['optimizer'])

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint['model'].items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v

# load params
model.load_state_dict(new_state_dict)

# Set the model to evaluation mode
model.eval()

# Inference on test images
for i in os.listdir("/kaggle/input/bkai-igh-neopolyp/test/test"):
    img_path = os.path.join("/kaggle/input/bkai-igh-neopolyp/test/test", i)
    ori_img = cv2.imread(img_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_w = ori_img.shape[0]
    ori_h = ori_img.shape[1]
    img = cv2.resize(ori_img, (256, 256))
    transformed = val_transformation(image=img)
    input_img = transformed["image"]
    input_img = input_img.unsqueeze(0).to(device)
    with torch.no_grad():
        output_mask = model.forward(input_img).squeeze(0).cpu().numpy().transpose(1,2,0)
    mask = cv2.resize(output_mask, (ori_h, ori_w))
    mask = np.argmax(mask, axis=2)
    mask_rgb = mask_to_rgb(mask, color_dict)
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite("prediction/{}".format(i), mask_rgb)

