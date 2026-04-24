/*
 * ================================================
 *  PC(Python)에서 시리얼로 문자 수신 → LED 제어
 *
 *  수신 명령어
 *    'G' → GREEN  ON  (정면 얼굴 감지)
 *    'Y' → YELLOW ON  (옆모습 감지)
 *    'R' → RED    ON  (얼굴 미감지 / 위험)
 *    'X' → 전체   OFF  (종료 / 초기화)
 * ================================================
 */

// ── 핀 정의 ──────────────────────────────────────
#define PIN_RED    9
#define PIN_YELLOW 10
#define PIN_GREEN  11

// ── 시리얼 통신 속도 (Python 쪽과 반드시 일치) ──
#define BAUD_RATE 9600

// ================================================
void setup() {
  // LED 핀 출력 모드 설정
  pinMode(PIN_RED,    OUTPUT);
  pinMode(PIN_YELLOW, OUTPUT);
  pinMode(PIN_GREEN,  OUTPUT);

  // 시작 시 전체 OFF (초기화)
  allOff();

  // 시리얼 통신 시작
  Serial.begin(BAUD_RATE);
  Serial.println("[LED Controller] Ready. Waiting for command (G/Y/R/X)...");

  // 시작 신호: 3색 순서대로 한 번씩 점등
  startupBlink();
}

// ================================================
void loop() {
  // 시리얼 수신 버퍼에 데이터가 있을 때만 처리
  if (Serial.available() > 0) {

    // 1바이트 읽기
    char cmd = Serial.read();

    // 명령어에 따라 LED 제어
    switch (cmd) {

      case 'G':   // 정면 감지 → GREEN
        allOff();
        digitalWrite(PIN_GREEN, HIGH);
        Serial.println("[LED] GREEN ON  — 정면 얼굴 감지");
        break;

      case 'Y':   // 옆모습 감지 → YELLOW
        allOff();
        digitalWrite(PIN_YELLOW, HIGH);
        Serial.println("[LED] YELLOW ON — 옆모습 감지");
        break;

      case 'R':   // 얼굴 미감지 → RED
        allOff();
        digitalWrite(PIN_RED, HIGH);
        Serial.println("[LED] RED ON    — 얼굴 미감지 (주의)");
        break;

      case 'X':   // 전체 OFF
        allOff();
        Serial.println("[LED] ALL OFF   — 초기화");
        break;

      default:
        // 알 수 없는 명령어는 무시 (개행문자 등 노이즈 방지)
        break;
    }
  }
}

// ================================================
// 함수: 전체 LED 소등
// ================================================
void allOff() {
  digitalWrite(PIN_RED,    LOW);
  digitalWrite(PIN_YELLOW, LOW);
  digitalWrite(PIN_GREEN,  LOW);
}

// ================================================
// 함수: 시작 시 3색 순서 점등 (배선 확인용)
//        RED → YELLOW → GREEN → OFF
// ================================================
void startupBlink() {
  digitalWrite(PIN_RED,    HIGH); delay(300);
  digitalWrite(PIN_RED,    LOW);
  digitalWrite(PIN_YELLOW, HIGH); delay(300);
  digitalWrite(PIN_YELLOW, LOW);
  digitalWrite(PIN_GREEN,  HIGH); delay(300);
  digitalWrite(PIN_GREEN,  LOW);
}
