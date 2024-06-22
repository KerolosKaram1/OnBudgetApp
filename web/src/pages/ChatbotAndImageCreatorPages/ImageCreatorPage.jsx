import styles from "../../styles/imageCreatorStyles/ImageCreator.module.css";
import ImgLeftSec from "../../components/ImageCreatorComponents/ImgLeftSec";
import ImgRightSec from "../../components/ImageCreatorComponents/ImgRightSec";

export default function ImageCreator() {
  return (
    <div className={styles.mainpage}>
      <div className={styles.leftOut}>
        <ImgLeftSec />
      </div>
      <div className={styles.rightOut}>
        <ImgRightSec />
      </div>
    </div>
  );
}
