shopt -s nullglob

declare -a filelist
for f in chi*.dat
do
    t=${f%.dat}
    filelist[${#filelist[@]}+1]=${t:9}
done
echo $filelist
IFS=$'\n' sorted=($(sort -g <<<"${filelist[*]}"));
unset IFS
echo "sorted:\n"
printf "[%s]\n" "${sorted[@]}"

i=0
mkdir -p chi_dir
for f in "${sorted[@]}"
do
    fn=`printf "chi_dir/chi%03d" $i`
    fn_old="chi0000_W${f}.dat"
    cut -d ' ' -f1,3 --complement "${fn_old}" | sed "/^$/d; s/^/${f} /g" | column -t -s ' ' > "${fn}"
    i=$[$i +1]
done
