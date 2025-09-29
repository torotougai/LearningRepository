# 命名の基本方針（最初に決め打ち）

* **言語別スタイル**

  * **Pythonコード**：`snake_case`、クラスは `PascalCase`、定数は `UPPER_SNAKE_CASE`
  * **URLパス**：`kebab-case`（例：`/user-profiles/{user_id}`）
  * **JSONキー**：`camelCase`（JS/TSと相性が良い）
  * **SQL（テーブル/列）**：`snake_case`（例：`user_profiles`, `created_at`）
* **単数/複数**

  * **クラス名・レコード型**は**単数**（`User`）
  * **リソース名/コレクション**は**複数**（`/users`、`users`テーブル、`users`配列）
* **略語・頭字語**

  * 許容：`id`, `url`, `http`, `api`, `ip`, `db`
  * クラスでは**語扱い**：`HttpClient` / 変数は小文字：`http_client`
  * 定数は上書きルール：`HTTP_TIMEOUT`
* **ブール名**：意味が読める接頭辞を強制
  `is_ / has_ / can_ / should_`（例：`is_active`, `has_permission`）
* **動詞・名詞**

  * **処理**＝動詞先頭（`create_user`, `fetch_report`）
  * **データ**＝名詞（`user`, `user_list`）

---

# Python（FastAPI/SQLAlchemy/Pydantic）

## モジュール/パッケージ/ファイル

* **パッケージ**：短い`lowercase`（必要なら`-`でなく`_`）例：`services`, `repositories`
* **モジュール（.py）**：`snake_case.py`（例：`user_service.py`）
* **テスト**：`tests/test_<module>.py`、クラスは`Test<Subject>`、関数は`test_<what>__<expectation>`

  * 例：`test_create_user__returns_201`

## クラス/例外/メンバ

* **クラス**：`PascalCase`（例：`UserService`, `UserRepository`）
* **例外**：`PascalCase`＋`Error`サフィックス（`UserNotFoundError`）
* **メソッド/関数**：`snake_case`。非同期は**サフィックスを付けない**（`async def fetch_user()`）
* **属性**：`snake_case`。内部用は`_`接頭辞（`_cache`）
* **ファクトリ/変換器**：`from_... / to_...`（`from_row`, `to_dict`）
* **イベント/ハンドラ**：`on_...`（`on_user_created`）

## レイヤ別の命名

* **Routerモジュール**：`users.py`（パスは`/users`）

  * エンドポイント関数：`list_users`, `get_user`, `create_user` などHTTP動詞に寄せた名詞併記
* **Service**：ビジネス動詞＋対象（`create_user`, `suspend_user`）
* **Repository**：`get_by_id`, `list`, `insert`, `update`, `delete` の固定語彙＋対象

## Pydantic スキーマ（JSONは camelCase）

* **命名**：`UserCreate`, `UserUpdate`, `UserRead`（入出力が即わかるサフィックス）
* **別名生成**（snake⇄camel を自動で橋渡し）

```python
# pydantic v2
from pydantic import BaseModel, ConfigDict

def to_camel(s: str) -> str:
    parts = s.split('_')
    return parts[0] + ''.join(p.title() for p in parts[1:])

class CamelBase(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

class UserRead(CamelBase):
    id: int
    display_name: str  # => JSONでは displayName
```

* **FastAPI**では`response_model_by_alias=True`でJSONをcamelに統一。

---

# URL/ルーティング

* **ベース**：`/users`, `/user-profiles`, `/auth-tokens`
* **kebab-case**の**名詞**で表現。動詞は使わない（`/create-user`はNG）
* **ID付き**：`/users/{user_id}`（パラメータ名は`snake_case`）
* **副作用のあるアクション**は**名詞＋サブリソース**：
  `POST /users/{user_id}/suspensions`, `POST /auth-tokens/refresh`
* **検索/フィルタ**：クエリは`snake_case`（`?page=1&per_page=50&created_before=...`）

---

# SQL / MySQL

## テーブル/列/制約

* **テーブル**：複数形`snake_case`（`users`, `user_profiles`）
* **列**：`snake_case`。外部キーは `<単数>_id`（`user_id`）
* **主キー**：`id`（BIGINT/AUTO_INCREMENT）
* **タイムスタンプ**：`created_at`, `updated_at`（UTC、アプリで管理）
* **ユニーク**：業務上ユニークは列名で分かる語を優先（`email` など）

## インデックス/制約の命名規約（SQLAlchemyで機械化）

* ルール：
  `pk_<table>` / `fk_<table>__<col>__<ref_table>` / `uq_<table>__<col...>` / `ix_<table>__<col...>`
* **SQLAlchemy MetaData**

```python
from sqlalchemy import MetaData

naming = {
    "ix":  "ix_%(table_name)s__%(column_0_label)s",
    "uq":  "uq_%(table_name)s__%(column_0_name)s",
    "ck":  "ck_%(table_name)s__%(constraint_name)s",
    "fk":  "fk_%(table_name)s__%(column_0_name)s__%(referred_table_name)s",
    "pk":  "pk_%(table_name)s",
}
metadata = MetaData(naming_convention=naming)
```

## マイグレーション

* **ファイル名**：`YYYYMMDDHHMMSS_<summary>.py`（例：`20250930_113000_add_user_profiles.py`）
* **リビジョン名**：`rev_<summary>`（英小文字と`_`のみ）

---

# 環境変数 / 設定

* すべて **UPPER_SNAKE_CASE**、サービス接頭辞を推奨：
  `APP_ENV`, `APP_LOG_LEVEL`, `DB_HOST`, `DB_USER`, `REDIS_URL`
* **秘密情報**は `SECRET_` 接頭辞で明示：`SECRET_JWT_KEY`
* **設定クラス**は `Settings`、属性は `snake_case`、PydanticでENV取込

---

# 「読みやすさを壊さない」細則

* **長さ**：25〜30字程度を目安。長いなら**語を減らす**か**分割**。
* **名詞の順序**：**コア名詞 → 修飾語**（`user_profile`, `report_monthly`）
* **否定形**は避ける：`is_disabled` より `is_active` の方が可読
* **集合は複数形**：`users`, `user_ids`
* **数えられない集合**：`inventory`, `equipment` のような英語の数不可算は文脈で配列かどうかを型で担保
* **重複語の削除**：`user.user_id` のような二度書きを避ける（カラムは `id`、外部キーは `user_id`）

---

# レイヤ横断の対応表（ズレゼロ設計）

| 概念     | URL                | JSON          | Python属性       | DB列            |
| ------ | ------------------ | ------------- | -------------- | -------------- |
| ユーザーID | `/users/{user_id}` | `userId`      | `user_id`      | `user_id`      |
| 表示名    | –                  | `displayName` | `display_name` | `display_name` |
| 作成日時   | –                  | `createdAt`   | `created_at`   | `created_at`   |

> 変換は **Pydanticのalias** で機械化。人間は常に「Python/DBは snake、JSONは camel」を書けばよい。

---

# 承認略語リスト（最少主義）

* **OK**：`id, url, uri, http, https, api, ip, db, ttl, jwt, oauth`
* **NG（具体語で）**：`cfg`→`config`, `svc`→`service`, `mgr`→`manager`, `tmp`→`temporary`

---

# 例：User を端から端まで

* **テーブル**：`users(id, email, display_name, created_at, updated_at)`
* **URL**：`GET /users/{user_id}`
* **Service**：`get_user(user_id: int) -> UserRead`
* **Repository**：`get_by_id(user_id: int) -> User`
* **Pydantic**：

  ```py
  class UserRead(CamelBase):
      id: int
      email: str
      display_name: str
      created_at: datetime
  ```
* **JSON**：

  ```json
  {
    "id": 1,
    "email": "alice@example.com",
    "displayName": "Alice",
    "createdAt": "2025-09-30T02:31:00Z"
  }
  ```

---

# 自動チェック（“破ったらCIが赤くなる”）

**pyproject.toml（抜粋）**

```toml
[tool.black]
line-length = 100
target-version = ["py311"]

[tool.ruff]
line-length = 100
extend-select = ["E", "F", "I", "UP", "B", "N"]  # N=pep8-naming
ignore = ["N818"]  # Exception命名強制は自前ルールで運用
[tool.ruff.isort]
known-first-party = ["app", "tests"]

[tool.pytest.ini_options]
python_classes = "Test*"
python_functions = "test_*"
```

**pre-commit（抜粋）**

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks: [{id: black}]
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks: [{id: ruff, args: ["--fix"]}]
```

**SQLAlchemy命名規約**は前掲の `naming_convention` を必ず適用。
**FastAPI**は `app = FastAPI(default_response_class=ORJSONResponse)` とし、`response_model_by_alias=True` をルーターやアプリのデフォルトに設定。

---

# 境界ケースへの指針

* **DTOの粒度**：入出力で別クラス（`UserCreate`と`UserRead`）。再利用で曖昧にしない。
* **ドメイン固有語**：まず**英語化の辞書**をリポジトリで管理（`docs/glossary.md`）。全員強制。
* **“一時”の名**は日付サフィックス禁止（`*_2025` など）。**意味**で名付ける。

---

# 運用の仕方（合意→強制）

1. この案を**チーム規約**として`docs/naming-conventions.md`に載せる
2. `pyproject.toml`と`pre-commit`を配布して**強制力**を持たせる
3. 既存コードは **変更触るときに同時リネーム**（Boy Scout Rule）
4. **用語辞書**を初期作成（20語程度）→PRで随時拡張
5. 新規APIはこの表に従って**URL/JSON/DBの対応表**をPRに必須化

---

このポリシーなら、「人が覚えるのは最小限」「機械が自動で揃える」が両立します。
採用であれば、**即使えるチーム向けドキュメント**と、`pyproject.toml`／`pre-commit-config.yaml`／Pydanticの`CamelBase`実装、SQLAlchemyの`naming_convention`を**そのまま貼れる形**で一式用意します。


いい命名はチームの“UI”です。読めば意味が一発で分かり、手が勝手に同じ型で書き始める──その状態を狙います。FastAPI＋Python（Pydantic/SQLAlchemy）＋MySQL前提で、**矛盾が出ない一貫ルール**をまとめました。

---

# 命名の基本方針（最初に決め打ち）

* **言語別スタイル**

  * **Pythonコード**：`snake_case`、クラスは `PascalCase`、定数は `UPPER_SNAKE_CASE`
  * **URLパス**：`kebab-case`（例：`/user-profiles/{user_id}`）
  * **JSONキー**：`camelCase`（JS/TSと相性が良い）
  * **SQL（テーブル/列）**：`snake_case`（例：`user_profiles`, `created_at`）
* **単数/複数**

  * **クラス名・レコード型**は**単数**（`User`）
  * **リソース名/コレクション**は**複数**（`/users`、`users`テーブル、`users`配列）
* **略語・頭字語**

  * 許容：`id`, `url`, `http`, `api`, `ip`, `db`
  * クラスでは**語扱い**：`HttpClient` / 変数は小文字：`http_client`
  * 定数は上書きルール：`HTTP_TIMEOUT`
* **ブール名**：意味が読める接頭辞を強制
  `is_ / has_ / can_ / should_`（例：`is_active`, `has_permission`）
* **動詞・名詞**

  * **処理**＝動詞先頭（`create_user`, `fetch_report`）
  * **データ**＝名詞（`user`, `user_list`）

---

# Python（FastAPI/SQLAlchemy/Pydantic）

## モジュール/パッケージ/ファイル

* **パッケージ**：短い`lowercase`（必要なら`-`でなく`_`）例：`services`, `repositories`
* **モジュール（.py）**：`snake_case.py`（例：`user_service.py`）
* **テスト**：`tests/test_<module>.py`、クラスは`Test<Subject>`、関数は`test_<what>__<expectation>`

  * 例：`test_create_user__returns_201`

## クラス/例外/メンバ

* **クラス**：`PascalCase`（例：`UserService`, `UserRepository`）
* **例外**：`PascalCase`＋`Error`サフィックス（`UserNotFoundError`）
* **メソッド/関数**：`snake_case`。非同期は**サフィックスを付けない**（`async def fetch_user()`）
* **属性**：`snake_case`。内部用は`_`接頭辞（`_cache`）
* **ファクトリ/変換器**：`from_... / to_...`（`from_row`, `to_dict`）
* **イベント/ハンドラ**：`on_...`（`on_user_created`）

## レイヤ別の命名

* **Routerモジュール**：`users.py`（パスは`/users`）

  * エンドポイント関数：`list_users`, `get_user`, `create_user` などHTTP動詞に寄せた名詞併記
* **Service**：ビジネス動詞＋対象（`create_user`, `suspend_user`）
* **Repository**：`get_by_id`, `list`, `insert`, `update`, `delete` の固定語彙＋対象

## Pydantic スキーマ（JSONは camelCase）

* **命名**：`UserCreate`, `UserUpdate`, `UserRead`（入出力が即わかるサフィックス）
* **別名生成**（snake⇄camel を自動で橋渡し）

```python
# pydantic v2
from pydantic import BaseModel, ConfigDict

def to_camel(s: str) -> str:
    parts = s.split('_')
    return parts[0] + ''.join(p.title() for p in parts[1:])

class CamelBase(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

class UserRead(CamelBase):
    id: int
    display_name: str  # => JSONでは displayName
```

* **FastAPI**では`response_model_by_alias=True`でJSONをcamelに統一。

---

# URL/ルーティング

* **ベース**：`/users`, `/user-profiles`, `/auth-tokens`
* **kebab-case**の**名詞**で表現。動詞は使わない（`/create-user`はNG）
* **ID付き**：`/users/{user_id}`（パラメータ名は`snake_case`）
* **副作用のあるアクション**は**名詞＋サブリソース**：
  `POST /users/{user_id}/suspensions`, `POST /auth-tokens/refresh`
* **検索/フィルタ**：クエリは`snake_case`（`?page=1&per_page=50&created_before=...`）

---

# SQL / MySQL

## テーブル/列/制約

* **テーブル**：複数形`snake_case`（`users`, `user_profiles`）
* **列**：`snake_case`。外部キーは `<単数>_id`（`user_id`）
* **主キー**：`id`（BIGINT/AUTO_INCREMENT）
* **タイムスタンプ**：`created_at`, `updated_at`（UTC、アプリで管理）
* **ユニーク**：業務上ユニークは列名で分かる語を優先（`email` など）

## インデックス/制約の命名規約（SQLAlchemyで機械化）

* ルール：
  `pk_<table>` / `fk_<table>__<col>__<ref_table>` / `uq_<table>__<col...>` / `ix_<table>__<col...>`
* **SQLAlchemy MetaData**

```python
from sqlalchemy import MetaData

naming = {
    "ix":  "ix_%(table_name)s__%(column_0_label)s",
    "uq":  "uq_%(table_name)s__%(column_0_name)s",
    "ck":  "ck_%(table_name)s__%(constraint_name)s",
    "fk":  "fk_%(table_name)s__%(column_0_name)s__%(referred_table_name)s",
    "pk":  "pk_%(table_name)s",
}
metadata = MetaData(naming_convention=naming)
```

## マイグレーション

* **ファイル名**：`YYYYMMDDHHMMSS_<summary>.py`（例：`20250930_113000_add_user_profiles.py`）
* **リビジョン名**：`rev_<summary>`（英小文字と`_`のみ）

---

# 環境変数 / 設定

* すべて **UPPER_SNAKE_CASE**、サービス接頭辞を推奨：
  `APP_ENV`, `APP_LOG_LEVEL`, `DB_HOST`, `DB_USER`, `REDIS_URL`
* **秘密情報**は `SECRET_` 接頭辞で明示：`SECRET_JWT_KEY`
* **設定クラス**は `Settings`、属性は `snake_case`、PydanticでENV取込

---

# 「読みやすさを壊さない」細則

* **長さ**：25〜30字程度を目安。長いなら**語を減らす**か**分割**。
* **名詞の順序**：**コア名詞 → 修飾語**（`user_profile`, `report_monthly`）
* **否定形**は避ける：`is_disabled` より `is_active` の方が可読
* **集合は複数形**：`users`, `user_ids`
* **数えられない集合**：`inventory`, `equipment` のような英語の数不可算は文脈で配列かどうかを型で担保
* **重複語の削除**：`user.user_id` のような二度書きを避ける（カラムは `id`、外部キーは `user_id`）

---

# レイヤ横断の対応表（ズレゼロ設計）

| 概念     | URL                | JSON          | Python属性       | DB列            |
| ------ | ------------------ | ------------- | -------------- | -------------- |
| ユーザーID | `/users/{user_id}` | `userId`      | `user_id`      | `user_id`      |
| 表示名    | –                  | `displayName` | `display_name` | `display_name` |
| 作成日時   | –                  | `createdAt`   | `created_at`   | `created_at`   |

> 変換は **Pydanticのalias** で機械化。人間は常に「Python/DBは snake、JSONは camel」を書けばよい。

---

# 承認略語リスト（最少主義）

* **OK**：`id, url, uri, http, https, api, ip, db, ttl, jwt, oauth`
* **NG（具体語で）**：`cfg`→`config`, `svc`→`service`, `mgr`→`manager`, `tmp`→`temporary`

---

# 例：User を端から端まで

* **テーブル**：`users(id, email, display_name, created_at, updated_at)`
* **URL**：`GET /users/{user_id}`
* **Service**：`get_user(user_id: int) -> UserRead`
* **Repository**：`get_by_id(user_id: int) -> User`
* **Pydantic**：

  ```py
  class UserRead(CamelBase):
      id: int
      email: str
      display_name: str
      created_at: datetime
  ```
* **JSON**：

  ```json
  {
    "id": 1,
    "email": "alice@example.com",
    "displayName": "Alice",
    "createdAt": "2025-09-30T02:31:00Z"
  }
  ```

---

# 自動チェック（“破ったらCIが赤くなる”）

**pyproject.toml（抜粋）**

```toml
[tool.black]
line-length = 100
target-version = ["py311"]

[tool.ruff]
line-length = 100
extend-select = ["E", "F", "I", "UP", "B", "N"]  # N=pep8-naming
ignore = ["N818"]  # Exception命名強制は自前ルールで運用
[tool.ruff.isort]
known-first-party = ["app", "tests"]

[tool.pytest.ini_options]
python_classes = "Test*"
python_functions = "test_*"
```

**pre-commit（抜粋）**

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks: [{id: black}]
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks: [{id: ruff, args: ["--fix"]}]
```

**SQLAlchemy命名規約**は前掲の `naming_convention` を必ず適用。
**FastAPI**は `app = FastAPI(default_response_class=ORJSONResponse)` とし、`response_model_by_alias=True` をルーターやアプリのデフォルトに設定。

---

# 境界ケースへの指針

* **DTOの粒度**：入出力で別クラス（`UserCreate`と`UserRead`）。再利用で曖昧にしない。
* **ドメイン固有語**：まず**英語化の辞書**をリポジトリで管理（`docs/glossary.md`）。全員強制。
* **“一時”の名**は日付サフィックス禁止（`*_2025` など）。**意味**で名付ける。

---

# 運用の仕方（合意→強制）

1. この案を**チーム規約**として`docs/naming-conventions.md`に載せる
2. `pyproject.toml`と`pre-commit`を配布して**強制力**を持たせる
3. 既存コードは **変更触るときに同時リネーム**（Boy Scout Rule）
4. **用語辞書**を初期作成（20語程度）→PRで随時拡張
5. 新規APIはこの表に従って**URL/JSON/DBの対応表**をPRに必須化

---