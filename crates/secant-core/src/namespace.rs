//! Schema-id and metadata-key namespace.
//!
//! Every schema id and metadata key Secant writes lives in the `secant.`
//! namespace (`secant.wsp.<name>.v1`, `secant.wsp.heal.*`, `secant.probe.*`).
//! On input, a legacy alias of the same id is also accepted so plans,
//! profiles and adapters written by older tools keep loading.

/// Namespace of every schema id and metadata key Secant writes.
pub const NAMESPACE: &str = "secant";

// Legacy alias accepted on input for files written by older tools.
pub const LEGACY_NAMESPACE: &str = "crucible";

/// The legacy alias of a canonical `secant.`-namespaced id, e.g.
/// `secant.wsp.plan.v1` -> `<legacy>.wsp.plan.v1`. `None` when `canonical`
/// is not in the `secant.` namespace (it then has no alias).
pub fn legacy_alias(canonical: &str) -> Option<String> {
    let rest = canonical.strip_prefix(NAMESPACE)?.strip_prefix('.')?;
    Some(format!("{}.{}", LEGACY_NAMESPACE, rest))
}

/// Input check: true when `found` is the canonical id `canonical` or its
/// legacy alias. Output must always use `canonical`.
pub fn accepts(canonical: &str, found: &str) -> bool {
    found == canonical || legacy_alias(canonical).as_deref() == Some(found)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::ANALYZE_SCHEMA;
    use crate::container::{INSPECT_SCHEMA, PATCH_SCHEMA, VERIFY_SCHEMA};
    use crate::heal::{HEAL_META_KEY, HEAL_META_PROTECTED_DIM, HEAL_META_PROTECTED_K, HEAL_SCHEMA, VERIFY_SUBSPACE_SCHEMA};
    use crate::plan::{COMPOSE_SCHEMA, PLAN_SCHEMA, PROFILES_SCHEMA, PROFILE_SCHEMA};
    use crate::read::READ_SCHEMA;
    use crate::security::{PROBE_ARCHITECTURE, PROBE_META_CLASS, PROBE_META_NUM_EXPERTS, PROBE_SCHEMA, SANITIZE_SCHEMA};

    fn all_written_ids() -> Vec<&'static str> {
        vec![
            PLAN_SCHEMA,
            PROFILE_SCHEMA,
            PROFILES_SCHEMA,
            ANALYZE_SCHEMA,
            COMPOSE_SCHEMA,
            READ_SCHEMA,
            INSPECT_SCHEMA,
            PATCH_SCHEMA,
            VERIFY_SCHEMA,
            HEAL_SCHEMA,
            VERIFY_SUBSPACE_SCHEMA,
            SANITIZE_SCHEMA,
            PROBE_SCHEMA,
            HEAL_META_KEY,
            HEAL_META_PROTECTED_DIM,
            HEAL_META_PROTECTED_K,
            PROBE_META_CLASS,
            PROBE_META_NUM_EXPERTS,
        ]
    }

    #[test]
    fn every_written_id_is_in_the_secant_namespace() {
        for id in all_written_ids() {
            assert!(id.starts_with("secant."), "{} is not written in the secant namespace", id);
            assert!(!id.contains(LEGACY_NAMESPACE), "{} carries the legacy namespace", id);
        }
        for id in [PLAN_SCHEMA, PROFILE_SCHEMA, COMPOSE_SCHEMA, READ_SCHEMA, INSPECT_SCHEMA, PATCH_SCHEMA, VERIFY_SCHEMA] {
            assert!(id.starts_with("secant.wsp.") && id.ends_with(".v1"), "{}", id);
        }
        assert_eq!(HEAL_META_KEY, "secant.wsp.heal");
        assert_eq!(HEAL_META_PROTECTED_DIM, "secant.wsp.heal.protected_dim");
        assert_eq!(HEAL_META_PROTECTED_K, "secant.wsp.heal.protected_k");
        assert_eq!(PROBE_META_CLASS, "secant.probe.class");
        assert_eq!(PROBE_META_NUM_EXPERTS, "secant.probe.num_experts");
        assert_eq!(PROBE_ARCHITECTURE, "secant-probe");
    }

    #[test]
    fn input_accepts_canonical_and_legacy_forms() {
        for id in all_written_ids() {
            let legacy = legacy_alias(id).expect("every written id has a legacy alias");
            assert_eq!(legacy, format!("{}{}", LEGACY_NAMESPACE, &id[NAMESPACE.len()..]));
            assert!(accepts(id, id), "canonical {} rejected", id);
            assert!(accepts(id, &legacy), "legacy {} rejected", legacy);
        }
        let legacy_plan = format!("{}.wsp.plan.v1", LEGACY_NAMESPACE);
        assert_eq!(legacy_alias("secant.wsp.plan.v1"), Some(legacy_plan.clone()));
        assert!(accepts(PLAN_SCHEMA, &legacy_plan));
        assert!(accepts(PLAN_SCHEMA, "secant.wsp.plan.v1"));
    }

    #[test]
    fn input_rejects_other_ids() {
        assert!(!accepts(PLAN_SCHEMA, PROFILE_SCHEMA));
        assert!(!accepts(PLAN_SCHEMA, "other.wsp.plan.v1"));
        assert!(!accepts(PLAN_SCHEMA, "secant.wsp.plan.v2"));
        assert!(!accepts(PLAN_SCHEMA, ""));
        assert!(!accepts(PLAN_SCHEMA, "wsp.plan.v1"));
        assert!(!accepts(PLAN_SCHEMA, &format!("{}.wsp.profile.v1", LEGACY_NAMESPACE)));
        assert!(!accepts(PLAN_SCHEMA, &format!("{}.wsp.plan.v2", LEGACY_NAMESPACE)));
        // a non-secant id has no alias and matches only itself
        assert_eq!(legacy_alias("secantx.wsp.plan.v1"), None);
        assert_eq!(legacy_alias("other.id"), None);
        assert!(accepts("other.id", "other.id"));
    }
}
