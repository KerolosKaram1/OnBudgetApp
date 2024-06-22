import styles from "../../styles/chatbotStyles/LeftSec.module.css";

function LeftSec() {
  return (
    <div className={styles.leftSection}>
      <div className={styles.logo}>
        <img src="/resources/images/logo1.png" alt="logo" />
      </div>
      <div className={styles.welcomeMessage}>
        <span>Welcome to Budget Bot assistant</span>
      </div>
    </div>
  );
}

export default LeftSec;
