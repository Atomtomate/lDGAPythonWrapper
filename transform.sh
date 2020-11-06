shopt -s nullglob

declare -a filelist
for f in chi*.dat; do
    t=${f%.dat}
    filelist[${#filelist[@]}+1]=${t:9}
done
IFS=$'\n' sorted=($(sort -g <<<"${filelist[*]}")); unset IFS

mkdir -p chi_dir
:> vert_chi
i=0
for f in "${sorted[@]}"; do
    fn=`printf "chi_dir/chi%03d" $i`
    fn_old="chi0000_W${f}.dat"
    cut -d ' ' -f1,3 --complement "${fn_old}" | sed "/^$/d; s/^/${f} /g" | awk '{printf ("%19.12f %19.12f %19.12f %19.12f %19.12f %19.12f %19.12f\n", $1, $2, $3, $4, $5, $6, $7)}'> "${fn}"
    cat "${fn}" >> vert_chi
    i=$[$i +1]
done
