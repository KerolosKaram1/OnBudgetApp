export function showPass(element) {
  if (element.current.type === "password") element.current.type = "text";
  else element.current.type = "password";
}
