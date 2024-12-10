# this is a script to download SMPL-X assest
# You need to login https://icon.is.tue.mpg.de/ and register SMPL-X and PIXIE

urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }


PROJ_DIR=$(pwd)
DOWNLOAD_PATH="extensions/pixielib/HPS/pixie_data"
mkdir -p ${DOWNLOAD_PATH}
echo -e "\nYou need to login https://pixie.is.tue.mpg.de/ &  https://smpl-x.is.tue.mpg.de/, and register SMPL-X and PIXIE"
read -p "Username (SMPL-X):" username
read -p "Password (SMPL-X):" password

username=$(urle $username)
password=$(urle $password)

# SMPLX Models
{
    wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=SMPLX_NEUTRAL_2020.npz&resume=1' -O ${DOWNLOAD_PATH}/SMPLX_NEUTRAL_2020.npz --no-check-certificate --continue
} || {
    wget https://huggingface.co/lilpotat/pytorch3d/resolve/346374a95673795896e94398d65700cb19199e31/SMPLX_NEUTRAL_2020.npz -O ${DOWNLOAD_PATH}/SMPLX_NEUTRAL_2020.npz  --continue
}
# PIXIE pretrained model and utilities
{
    wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=pixie&sfile=pixie_model.tar&resume=1' -O ${DOWNLOAD_PATH}/pixie_model.tar --no-check-certificate --continue
} || {
    wget https://huggingface.co/lilpotat/pytorch3d/resolve/main/pixie_model.tar -O ${DOWNLOAD_PATH}/pixie_model.tar  --continue
}
cd ${DOWNLOAD_PATH} && tar xvf utilities.tar
cd $PROJ_DIR










