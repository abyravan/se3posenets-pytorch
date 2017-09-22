FOLDER_PATH=$1
if [ ! -z $2 ] 
then 
    FRAME_RATE=$2 # $2 was given
else
    FRAME_RATE=30 # $2 was not given
fi
if [ ! -z $3 ] 
then 
    VIDEO_NAME=$3 # $3 was given
else
    VIDEO_NAME='video' # $3 was not given
fi
echo 'Looking for files in folder: '$FOLDER_PATH
echo 'Frame rate: '$FRAME_RATE
echo 'Saving video at: '$FOLDER_PATH'/'$VIDEO_NAME'.mp4'


ffmpeg -r $FRAME_RATE \
    -start_number 1 -i $FOLDER_PATH'/render%d.png' -c:v libx264 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -pix_fmt yuv420p \
    $FOLDER_PATH'/'$VIDEO_NAME'.mp4'

