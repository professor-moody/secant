use secant_core::rotation::{rotation_test, RotationOptions};

#[test]
fn deterministic_three_arm_witness_and_adamw_rotation() {
    let w: Vec<f64> = (0..48).map(|i| ((i * 17 % 29) as f64 - 14.0) / 7.0).collect();
    let d = vec![vec![1.0 / 14f64.sqrt(), 2.0 / 14f64.sqrt(), 3.0 / 14f64.sqrt(), 0.0, 0.0, 0.0]];
    for wd in [0.0, 0.01] {
        let o = RotationOptions { rank: 2, steps: 12, lr: 0.01, wd, seed: 7 };
        let r = rotation_test(&w, 6, 8, &d, &o).unwrap();
        assert!(r.lock_passed);
        assert!(r.arms[0].step_1.relative_leakage > 0.01);
        assert!(r.arms[1].step_1.relative_leakage > 0.01, "projecting gradients does not constrain diagonal AdamW preconditioning");
        assert!(r.arms[2].step_t.relative_leakage < 1e-12);
        assert!(r.arms[2].step_t.loss + 1e-12 >= r.locked_loss_floor);
        assert!(r.arms.iter().all(|a| a.step_t.loss < a.trajectory[0].loss));
        assert_eq!(serde_json::to_string(&r).unwrap(), serde_json::to_string(&rotation_test(&w, 6, 8, &d, &o).unwrap()).unwrap());
    }
}

#[test]
fn invalid_inputs_and_vacuous_updates_are_not_passes() {
    let o = RotationOptions { rank: 1, steps: 1, lr: 0.01, wd: 0.0, seed: 0 };
    assert!(rotation_test(&[0.0; 4], 2, 2, &[vec![1.0, 0.0]], &o).is_err());
    assert!(rotation_test(&[f64::NAN; 4], 2, 2, &[vec![1.0, 0.0]], &o).is_err());
    assert!(rotation_test(&[1.0; 4], 2, 2, &[vec![2.0, 0.0]], &o).is_err());
    assert!(rotation_test(&[1.0; 4], 2, 2, &[vec![1.0, 0.0]], &RotationOptions { steps: 0, ..o.clone() }).is_err());
    let r = rotation_test(&[1.0, 2.0, 0.0, 0.0], 2, 2, &[vec![1.0, 0.0]], &o).unwrap();
    assert!(!r.lock_passed);
    assert!(r.arms[2].step_t.delta_is_zero);
}
