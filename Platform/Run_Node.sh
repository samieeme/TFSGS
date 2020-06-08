#! /bin/bash
cd $1 ## Data file
REALIZATION=$(ls)
for item in ${REALIZATION}
do
  cd ${item}
  echo "items = ${item}"
  DECAYRE=$(ls -d */)
  for item2 in ${DECAYRE%%/}
  do
    cd $1/${item}/${item2} 
    HOMETMP=$(pwd)
    FILE_NAME=$(ls *.mat)
    ADD_OUT="flt_DNS"
    echo ${HOMETMP}
    cd $2
    if [ -e ${item} ]
    then
      echo "it exists"
    #  rm -rf $2/${item}/*
    else
      mkdir -p ${item}
    fi
    #echo ${FILE_NAME}
    cd $2/${item}
    OUT_FLT_DATA=$(pwd)
    python /mnt/scratch/samieeme/TFSGS/Filtering/main.py $HOMETMP $FILE_NAME $OUT_FLT_DATA
    FILE_NAME=$(ls *) 
    cd $3
    if [ ! -e ${item} ]
    then
      mkdir $3/${item}
    fi
    cd $3/${item}
    if [ ! -e ${item2} ]
    then
      mkdir $3/${item}/${item2}
    fi
    rm -rf $3/${item}/${item2}/*
    cd ./${item2}/
    FILEOUT=$(pwd)
    echo $FILEOUT
    echo $OUT_FLT_DATA
    echo $FILE_NAME
    python /mnt/scratch/samieeme/TFSGS/LES_I/TFSGS.py 5 $OUT_FLT_DATA $FILE_NAME $FILEOUT  
  done                                   
  #echo ${HOMETMP}  
  #python /mnt/scratch/samieeme/TFSGS/Filtering/main.py $HOMETMP $FILE_NAME $OUT_FLT_DATA  
  cd $1  
done     

