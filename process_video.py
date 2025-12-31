import cv2
import numpy as np
import os

SAVE_DENOISED = True  # Set to True to save denoised binary images
SAVE_CONTOURS = True  # Set to True to save contour images
SAVE_CONTOURS_DOWNSAMPLED = True  # Set to True to save downsampled contour images
OUTPUT_FPS = 1.2  # output frames per second


def process_frames(video_path, frames_path, fps): # extract 1 frame per second
    frame_duration_ms = int(1000 / OUTPUT_FPS)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_ms = (total_frames / fps) * 1000 if fps > 0 else 0
    
    i = 0 # frame counter
    # from 0 seconds to duration, step 1000 ms
    for time_ms in range(0, int(duration_ms), frame_duration_ms): 
        cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(frames_path, f"frame_{i}.jpg"), frame)
            print(f"Extracted frame {i} at {time_ms} ms")
        i += 1

    cap.release()
    cv2.destroyAllWindows()


def detect_changes(column, direction): # returns the changes, 0 at top of column. Change occurs when pixel changes shade
    changes = []
    prev_pixel = column[0]
    for y in range(1, len(column)):
        if column[y] != prev_pixel:
            changes.append(y)
        prev_pixel = column[y]
        
    if len(changes) == 0:
        if direction == 'up':
            changes.append(len(column)-1)
        else:
            changes.append(0)
        
    return changes


def find_coutours(frame):
    contourA = []
    contourB = []
    
    for col_idx in range(frame.shape[1]):
        
        changes = detect_changes(frame[:, col_idx], 'down')
        contourA.append(changes[0])
        
        changes = detect_changes(frame[:, col_idx], 'up')
        contourB.append(changes[-1])
    
    return contourA, contourB


def contour_images(path, save_denoised=False, save_contours=False, save_contours_downsampled=False):
    # Remove existing tracedata.txt
    # File stores the HPIB commands, each frame on a new line
    command_file_path = os.path.join(path, 'commands.txt')
    if os.path.exists(command_file_path):
        os.remove(command_file_path)
    
    # Get list of frame images, sorted by frame number assumed in filename after underscore
    img_list = [f for f in os.listdir(path) if f.endswith('.jpg') and '_contours' not in f and '_denoise' not in f]
    img_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    for img_path in img_list:
        print(f"Processing {img_path}...")
        # Read image and convert to binary
        frame = cv2.imread(os.path.join(path, img_path))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV) # for some reason THRESH_BINARY_INV works better here
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)        
        
        # Find contours
        top_contours, bottom_contours = find_coutours(binary)
        
        # Transform contours to desired range and downsample to correctly fit to SA display
        transformed_top = transform_contour(top_contours, (0, binary.shape[0]-1), (1, 975), 481)
        transformed_bottom = transform_contour(bottom_contours, (0, binary.shape[0]-1), (1, 975), 481)
        with open(command_file_path, 'a') as f:
            # Each line contains IA (top contour) and IB (bottom contour) commandse
            f.write('IA' + ','.join(map(str, transformed_top)) + ';IB' + ','.join(map(str, transformed_bottom)) + ';\n')
        
        # Save diagnostic images if enabled
        if save_denoised: 
            cv2.imwrite(os.path.join(path, img_path.replace('.jpg', '_denoise.jpg')), binary)
        if save_contours:
            # create blank image for contours
            contour_img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            # plot contours on blank image
            for x in range(len(top_contours)):
                cv2.circle(contour_img, (x, top_contours[x]), 1, (0, 0, 255), -1)
                cv2.circle(contour_img, (x, bottom_contours[x]), 1, (0, 255, 0), -1)
            cv2.imwrite(os.path.join(path, img_path.replace('.jpg', '_contours.jpg')), contour_img)
        if save_contours_downsampled:
            # Create a blank image with a 4:3 aspect ratio
            new_width = 481  # Horizontal axis has 481 points
            new_height = int(new_width * 3 / 4)  # To maintain a 4:3 aspect ratio

            # Create an empty image with the correct dimensions (361, 481)
            contour_img_ds = np.zeros((new_height, new_width, 3), dtype=np.uint8)

            # Iterate over the transformed contours and plot them in the new image
            for x in range(len(transformed_top)):
                # Scale the top and bottom contours to the new vertical range (1 to 975) to (1 to 361)
                y_top = int((new_height - 1) - (transformed_top[x] - 1) * (new_height - 1) / 975)
                y_bottom = int((new_height - 1) - (transformed_bottom[x] - 1) * (new_height - 1) / 975)

                # Plot the contours on the downsampled image
                cv2.circle(contour_img_ds, (x, y_top), 1, (0, 0, 255), -1)  # Red for top contour
                cv2.circle(contour_img_ds, (x, y_bottom), 1, (0, 255, 0), -1)  # Green for bottom contour

            # Save the downsampled contour image
            cv2.imwrite(os.path.join(path, img_path.replace('.jpg', '_contours_downsampled.jpg')), contour_img_ds)

    
def transform_contour(cont, old_vertical_range, new_vertical_range, n_points=481): 
    NewRange = (new_vertical_range[1]-new_vertical_range[0])    
    cont = [int(new_vertical_range[1] - (i - old_vertical_range[0]) * NewRange / (old_vertical_range[1] - old_vertical_range[0])) for i in cont]
    
    if len(cont) > n_points:
        step = len(cont) / n_points
        cont = [cont[int(i * step)] for i in range(n_points)]  # take every step, up to n
    return cont


if __name__ == "__main__":
    dir = './data/bad_apple/'
    if os.path.exists(dir) == False:
        os.makedirs(dir)

    # Extract frames from video
    process_frames('bad_apple.mp4', dir, OUTPUT_FPS)
    
    # Generate contour commands from frames
    contour_images(dir, SAVE_DENOISED, SAVE_CONTOURS, SAVE_CONTOURS_DOWNSAMPLED)
