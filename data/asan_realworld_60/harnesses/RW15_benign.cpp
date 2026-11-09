#include <optional>
#include <string>
#include <vector>

struct RuleEntry { const char* name; };
struct GrammarBuilder {
  std::vector<RuleEntry> rules;
  int BuildRegexFSM(const std::string& regex, const char* hint) {
    (void)hint;
    for (int i = 0; i < 16; ++i) rules.push_back(RuleEntry{"aux"});
    return regex == "bad" ? -1 : 0;
  }
};

struct GrammarFSMBuilderImpl {
  GrammarBuilder* builder_;
  const RuleEntry* rule_name_;
  std::optional<std::string> rule_name_copy_;

  int BuildRegex(const std::string& regex, std::string* out_err) {
    if (!rule_name_copy_.has_value() && rule_name_ != nullptr) {
      rule_name_copy_ = rule_name_->name;
    }
    const char* rule_hint = rule_name_copy_.has_value() ? rule_name_copy_->c_str() : "";
    int rc = builder_->BuildRegexFSM(regex, rule_hint);
    if (rc < 0 && rule_name_copy_.has_value()) {
      *out_err = *rule_name_copy_;
    }
    return rc;
  }
};

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  GrammarBuilder gb;
  gb.rules.push_back(RuleEntry{"root"});
  GrammarFSMBuilderImpl fsm{&gb, &gb.rules[0], std::nullopt};
  std::string err;
  int rc = fsm.BuildRegex("bad", &err);
  volatile size_t keep = err.size() + static_cast<size_t>(rc);
  (void)keep;
  return 0;
}
