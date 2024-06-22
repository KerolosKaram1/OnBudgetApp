import InfoIcon from "@mui/icons-material/Info";
import "../../styles/utilsStyles/Alert.css";

export default function SimpleAlert({ message, msgColor }) {
  return (
    <div className="alert-message" style={{ backgroundColor: msgColor }}>
      <InfoIcon style={{ marginRight: 5 }} />
      {message}
    </div>
  );
}
