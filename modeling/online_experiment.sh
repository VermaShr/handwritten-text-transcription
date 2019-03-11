outfolder=$1 #./tf_output/online_training/
infolder=$2 #./tf_output/official_training/
n_batches=$3

rm -r $outfolder
mkdir -p $outfolder
cp $2* $outfolder

#create asm batch
cd ../preprocess
python create_ASM_batch.py 10 10 0.5 False
cd ../modeling
#predict new data
python run_model.py pred ASM $n_batches 10 True 1000 $outfolder new $outfolder 4
#train new data
# python run_model.py train ASM 2 16 True 1000 $outfolder new $outfolder 1000
#
# #create random batch
# python create_ASM_batch.py 1000 1000 0.5 ../data True
# #train old data
# python run_model.py train ASM 2 16 True 1000 $outfolder old $outfolder 1000




# for i in {2000..50000..1000}
#   do
#     #create asm batch
#     cd ../preprocess
#     python create_ASM_batch.py $i 1000 0.5 ../data False
#     cd ../modeling
#     # predict new data
#     python run_model.py pred ASM 1 16 True $i $outfolder new $outfolder $(($i-1000))
#     if [[ $? -ne 0 ]]; then
#     	break
#     fi
#
#     #train new data
#     python run_model.py train ASM 2 16 True $i $outfolder new $outfolder $i
#     if [[ $? -ne 0 ]]; then
#     	break
#     fi
#     # create random batch
#     python create_ASM_batch.py $i 1000 0.5 ../data True
#     if [[ $? -ne 0 ]]; then
#     	break
#     fi
#     # train old data
#     python run_model.py train ASM 2 16 True $i $outfolder old $outfolder $i
#     if [[ $? -ne 0 ]]; then
#     	break
#     fi
#     #delete old files
#     python end_batch.py $i $outfolder 2000
#   done
