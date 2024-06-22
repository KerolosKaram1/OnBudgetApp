import React from "react";
import "./styles/Sections.css";

export default function Section_1({ section, setSection }) {
  return (
    <section className="section_1">
      <div className="left-side">
        <h1>
          Welcome
          <span role="img" aria-label="emoji">
            🥰
          </span>
        </h1>

        <p>
          Every thing you need in one place & <span>onBudget</span>.
          <br />
          The easiest way to explore the world of fashion.
        </p>

        <button onClick={() => setSection("section_2")}>Next</button>

        <div className="dots">
          <div
            className="dot1"
            style={{ backgroundColor: "rgb(255, 205, 78)" }}
          ></div>
          <div className="dot2"></div>
          <div className="dot3"></div>
        </div>
      </div>

      <div className="right-side">
        <img
          src={"resources/images/svg/welcome.svg"}
          alt="welcome"
          loading="lazy"
        />
      </div>
    </section>
  );
}
