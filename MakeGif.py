import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import image

plt.rcParams['figure.dpi'] = 300
plt.rcParams['animation.convert_path'] = r'C:\Program Files\ImageMagick-7.0.9-Q16\magick.exe'


def GetTimeHourUTC(some_str: str) -> float:
    idx_timestart = some_str.find('_') + 1
    hh = some_str[idx_timestart:idx_timestart + 2]
    mm = some_str[idx_timestart + 2:idx_timestart + 4]
    ss = some_str[idx_timestart + 4:idx_timestart + 6]
    time_hour = float(hh) + float(mm) / 60 + float(ss) / 3600
    return time_hour


def MakeVideo(batch_folder, day_folder, radar_product, start_time_hour, stop_time_hour, save_movie=False):
    radar_product_dir = os.path.join(batch_folder, day_folder, radar_product)
    output_dir = os.path.join(batch_folder, day_folder, "videos")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    picFiles = os.listdir(radar_product_dir)
    picFiles_filtered = []
    for pic in picFiles:
        curr_time = GetTimeHourUTC(pic)
        if curr_time >= start_time_hour and curr_time <= stop_time_hour:
            picFiles_filtered.append(pic)

    # Set up formatting for the movie files
    Writer = animation.writers['imagemagick']
    writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800)

    fig = plt.figure()

    ims = []
    for pic in picFiles_filtered:
        print(pic)

        X = image.imread(os.path.join(radar_product_dir, pic))
        plt.axis('off')
        im = plt.imshow(X)
        plt.tight_layout(pad=0.3)
        ims.append([im])

    # animate movie
    outFile = "{}_{}_{}.gif".format(radar_product, start_time_hour, stop_time_hour)
    anim = animation.ArtistAnimation(fig, ims, interval=450, blit=True, repeat_delay=1)

    if save_movie:
        anim.save(os.path.join(output_dir, outFile), writer=writer, dpi=400)
    plt.show()
    plt.close()
    print("The end ...")


def Main():
    batch_folder = "./figures/KOHX_20180501_20180515"
    day_folder = "KOHX20180501"
    radar_product = "differential_phase"
    start_time_hour = 11
    stop_time_hour = 13
    save_movie = True

    MakeVideo(batch_folder, day_folder, radar_product, start_time_hour, stop_time_hour, save_movie)


Main()
