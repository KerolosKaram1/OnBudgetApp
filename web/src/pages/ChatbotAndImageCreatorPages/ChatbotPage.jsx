import styles from "../../styles/chatbotStyles/Chatbot.module.css";
import LeftSec from "../../components/ChatbotComponents/LeftSec";
import RightSec from "../../components/ChatbotComponents/RightSec";

export default function Chatbot() {
  return (
    <div className={styles.mainpage}>
      <div className={styles.leftOut}>
        <LeftSec />
      </div>
      <div className={styles.rightOut}>
        <RightSec />
      </div>
    </div>
  );
}
