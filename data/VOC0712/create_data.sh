# 注意，使用绝对路径
# dataset的父目录 
data_root_dir="/homec/wyj/dataset/VOCdevkit/VOC2012/"
# create_list.py生成的标注文件
list_file="/homec/wyj/dataset/VOCdevkit/RefineDet/trainval.txt"
# lmdb存放的目录
outdir="/homec/wyj/dataset/VOCdevkit/RefineDet/lmdb"

# 标签datum
map_file="labelmap_voc.prototxt"

# the script file: 待调用的脚本(注意还要修改这个脚本里面的Python path, 添加refineDet的python,让它能找到caffe包)
create_annoset="/homec/wyj/github/RefineDet/scripts/create_annoset.py"

# python interpreter: 使用哪个解释器
my_python="/homec/wyj/anaconda2/bin/python"

# need not modify
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0
redo=1

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi

$my_python $create_annoset --anno-type=$anno_type --label-map-file=$map_file --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $list_file $outdir

