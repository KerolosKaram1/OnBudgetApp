export function selectMale(male, female, setUserGender) {
  if (male.current.style.border === "4px solid rgb(255, 205, 78)") {
    male.current.style.border = "none";
    setUserGender("");
  } else {
    male.current.style.border = "4px solid rgb(255, 205, 78)";
    female.current.style.border = "none";
    setUserGender("male");
  }
}

export function selectFemale(male, female, setUserGender) {
  if (female.current.style.border === "4px solid rgb(255, 205, 78)") {
    female.current.style.border = "none";
    setUserGender("");
  } else {
    female.current.style.border = "4px solid rgb(255, 205, 78)";
    male.current.style.border = "none";
    setUserGender("female");
  }
}
