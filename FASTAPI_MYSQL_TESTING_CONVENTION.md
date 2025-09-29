# FastAPI＋MySQL プロジェクト共通テスト規約（ユニット/契約/統合）

> 目的：初心者でも迷わず、保守性と開発速度を両立する。“速く回るユニット”を主戦場に、API契約とDB統合は薄く強く。FastAPI と MySQL を前提に、pytest を標準化します。

---

## 1. 原則（守るほど速く・壊れにくい）

* **純粋ロジック最優先**：I/O（DB/HTTP/ファイル/時刻/乱数）を切り離したユニットテストが主戦場。
* **契約指向**：FastAPI の入出力は DTO（Pydantic）とステータスの“契約”で検証。実装詳細には依存しない。
* **DBは統合へ隔離**：MySQL を触るのは `integration` マークの統合テストのみ。ユニットは I/O ゼロ。
* **小さく速く独立**：1テスト＝1主張。順序依存を作らない。10〜50ms/ケースを目安。
* **再現→修正**：不具合は必ず再現テストを先に追加してから修正。

---

## 2. テストの粒度とカバレッジ目標

### 2.1 粒度レベル

1. **Unit（ユニット）**

   * 対象：ドメイン/ユースケースの純ロジック、アダプタの引数検証、ユーティリティ。
   * 依存：DB/HTTP/時刻/UUIDは**ポート化**してフェイク注入。
   * 目標：全体の **70% 以上** を占める。1秒で100本回る規模を維持。
2. **API Contract（契約）**

   * 対象：FastAPI ルータ・依存関係の結線・DTOのバリデーション・HTTP ステータス／本文の整合。
   * 手法：`TestClient` + `dependency_overrides` で**フェイク**を注入し、I/Oゼロで実行。
3. **Integration（統合：MySQL）**

   * 対象：SQLAlchemy/Alembic マイグレーション、Repository の実 SQL、トランザクション、インデックス、一意制約、N+1 検知。
   * 手法：テスト用 DB（Docker）で**トランザクション巻き戻し**（各テスト終了時 rollback）。

### 2.2 カバレッジ（数値は“意味づけ”して使う）

* 行カバレッジ **80%** 以上、ブランチ **60%** 以上。
* クリティカルユースケース（課金・認証等）は **95%** 目標。
* カバレッジはゲートだが、落ちたら**除外対象**（設定・生成物）を定期見直し。

---

## 3. ディレクトリ構成と命名

```
app/
  domain/            # 純ロジック（副作用ゼロ）
  usecases/          # 入出力DTOとユースケース調停
  adapters/          # Repo, 外部HTTPクライアント（抽象＋実装）
  api/               # FastAPI ルータ（薄皮）

tests/
  unit/
    domain/
    usecases/
    adapters/
    api_contract/
    _builders/       # テストデータビルダ
    _fakes/          # フェイク実装（In-memory）
    _helpers/        # 共通アサーションや固定Clock等
  integration/
    db/
    migrations/

pytest.ini
```

**命名規則**：`test_<機能>__<条件>__<期待>()`（アンダースコア2本で読みやすく検索しやすい）。

---

## 4. PyTest の使い方（パラメトライズ中心）

### 4.1 パラメトライズ（境界値と代表値を“表”に）

```python
import pytest

@pytest.mark.unit
@pytest.mark.parametrize(
    "password, expected",
    [
        ("aB3!xxxx", True),           # 最小長境界を満たす
        ("short1!", False),           # 長さ不足
        ("NoSpecial1234", False),    # 記号なし
        ("汉字ABC123!", True),        # マルチバイト混在
    ],
)
def test_password_policy__cases(password, expected, policy):
    assert policy.validate(password) is expected
```

* **原則**：1テスト＝1関数で**複数ケース**を表にする（`parametrize`）。
* **ID付与**：可読性のため `ids=["min_ok","too_short","no_symbol","multibyte"]` を活用。
* **例外系**：`pytest.param(..., marks=pytest.mark.xfail(reason="..."))` で既知欠陥を可視化。

### 4.2 フィクスチャの方針

* 原則は**明示注入**。巨大 `conftest` の暗黙 `autouse` は多用しない。
* 代表例：`fixed_clock`, `seq_uuid`, `fake_user_repo`, `client_with_fakes`。

```python
@pytest.fixture
def fixed_clock():
    return FixedClock("2025-01-01T00:00:00Z")
```

### 4.3 アサーションの読みやすさ

* `assert expr, "なぜ失敗か"` を徹底。共通は `tests/_helpers/asserts.py` に関数化。

---

## 5. テストダブル（モック/スタブ/フェイク）運用

* **戻り値が決定的**：スタブ（簡単）。
* **相互作用を検査したい**：最小限のモック（回数検証の乱用禁止）。
* **データストアの挙動**：インメモリ **フェイク**（ID採番・一意制約・トランザクション疑似）。

```python
class FakeUserRepo(UserRepo):
    def __init__(self):
        self._store = {}
    def add(self, user: User):
        if user.email in self._store:
            raise ConflictError("email")
        self._store[user.email] = user
```

---

## 6. FastAPI 向けルール（契約テスト）

* ルータは**薄く**：DTO変換→ユースケース呼出→HTTP へ写像。
* **依存差し替え**：`app.dependency_overrides[get_user_repo] = lambda: FakeUserRepo()`
* **契約テスト**：`TestClient` で I/O なしの HTTP 振る舞いを確認。

```python
from fastapi.testclient import TestClient

@pytest.mark.unit
def test_register__happy_path(client_with_fakes):
    res = client_with_fakes.post("/users", json={"email": "a@b.c", "password": "aB3!xxxx"})
    assert res.status_code == 201
    assert res.json()["email"] == "a@b.c"
```

* **バリデーション**：リクエスト/レスポンスモデルは Pydantic で厳密化。テストは**ステータス**と**スキーマ**を主に検査（メッセージ文言には依存しない）。

---

## 7. MySQL 統合テスト方針

* **環境**：Docker で `mysql:8`（`utf8mb4_0900_ai_ci`）。`docker compose -f compose.test.yml up -d`。
* **接続**：SQLAlchemy (sync/async はプロジェクトに合わせる)。
* **マイグレーション**：Alembic を必ず経由。テストで `alembic upgrade head` → ケースごとに **BEGIN/ROLLBACK**。
* **データ投入**：Factory/Fixture で最小限（1テスト1〜3行）。ファイルで大量投入しない。
* **N+1 検知**：`sqlalchemy` の `echo` + クエリカウントアサート（helpers で共通化）。

```python
@pytest.mark.integration
def test_user_repo__unique_email(db_session, user_factory):
    repo = SqlUserRepo(db_session)
    repo.add(user_factory(email="x@y.z"))
    with pytest.raises(ConflictError):
        repo.add(user_factory(email="x@y.z"))
```

---

## 8. 認証/2FA（TOTP）テスト基準

* **時刻固定**：`FixedClock` で `now()` を固定。±1ステップのスキューをパラメトライズ。
* **ハッシュ/署名**：`FakeHasher`/`FakeSigner` で決定的に。JWT は exp/aud/scope を表駆動で検査。
* **ロックアウト**：連続 N 回の閾値、クールダウン境界（直前/直後）を表にする。

```python
@pytest.mark.parametrize("code, skew, ok", [
    ("123456", 0, True),
    ("123456", +1, True),
    ("123456", -2, False),
])
```

---

## 9. 速度最適化（DXを壊さない運用）

* **マーカー分離**：`unit` と `integration`。ローカル `pre-commit` は `unit` のみ。
* **並列化**：CI は `pytest -n auto`（`pytest-xdist`）。
* **影響範囲実行**：変更ファイルに応じたディレクトリ単位のセレクティブ実行（できれば TIA を導入）。
* **キャッシュ**：依存解決・`__pycache__`・`pip`/`venv`・`.pytest_cache` を CI でキャッシュ。

**目安**：ユニット 500 本で < 10 秒、統合 100 本で < 2 分。

---

## 10. 品質ゲートと PR チェックリスト

* [ ] 新規ユースケースは **ユニット**同梱
* [ ] ルータ追加/DTO変更は **契約テスト**更新
* [ ] Repository 変更は **統合テスト**で SQL が通ること
* [ ] **カバレッジ閾値**を満たす（行80/分岐60）
* [ ] 大きなフィクスチャや暗黙依存はない
* [ ] 失敗時メッセージが読める（何が、なぜ、どう期待）

---

## 11. アンチパターン（やらない）

* DB を触るロジックをユニットに持ち込む
* `sleep` で非決定性を誤魔化す
* 文言（i18n）にアサートを縛る
* 1 テストに複数の主張
* 巨大な `conftest.py` に暗黙 `autouse=True` を散布

---

## 12. 運用開始手順（最初の 1〜2 日）

1. `Clock`/`IdGenerator` など副作用の**ポート化**。
2. `tests/unit/_fakes` と `/_builders` の最小セット（User/Token/Repo）を作成。
3. 代表ユースケースを**表駆動テスト**（`parametrize`）化。
4. FastAPI の 1 エンドポイントを**契約テスト**化（依存差し替え）。
5. Docker MySQL + Alembic で **ROLLBACK 方式**の統合テストを 1 本通す。

---

## 13. 付録：pytest 実行コマンド例

```bash
# 速いユニットだけ
pytest -m unit -q

# すべて（並列）
pytest -n auto -q

# カバレッジ付き
pytest -m unit --cov=app --cov-report=term-missing

# 統合だけ（DB起動済み前提）
pytest -m integration -q
```

---

## 14. 付録：よく使うヘルパ（骨子）

```python
# tests/_helpers/clock.py
class FixedClock:
    def __init__(self, iso: str):
        self._now = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    def now(self) -> datetime:
        return self._now

# tests/_builders/user.py
class UserBuilder:
    def __init__(self):
        self.email = "user@example.com"
        self.twofa = False
    def with_email(self, email):
        self.email = email; return self
    def with_2fa(self):
        self.twofa = True; return self
    def build(self):
        return User(email=self.email, twofa=self.twofa)

# tests/_helpers/asserts.py
def assert_http(res, status):
    assert res.status_code == status, f"status={res.status_code} body={res.text}"
```

---

### この規約は“生き物”です

運用で得た知見を3ヶ月に1度レビューし、規約を更新しましょう。テストは筋トレと同じで、正しいフォームが最速の近道です。