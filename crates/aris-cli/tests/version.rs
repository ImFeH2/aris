use std::process::Command;

#[test]
fn prints_version() {
    let output = Command::new(env!("CARGO_BIN_EXE_aris"))
        .arg("--version")
        .output()
        .expect("failed to run aris");

    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).expect("stdout is not valid UTF-8"),
        format!("aris {}\n", env!("CARGO_PKG_VERSION"))
    );
}
