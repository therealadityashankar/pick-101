// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import starlightTypeDoc, { typeDocSidebarGroup } from "starlight-typedoc";

// https://astro.build/config
export default defineConfig({
  site: "https://therealadityashankar.github.io",
  base: "/pick-101",
  integrations: [
    starlight({
      title: "lerobot-tagged",
      description:
        "ArUco board generation and tag-based robot arm localisation — browser, Node, and Python",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/therealadityashankar/pick-101",
        },
      ],
      plugins: [
        starlightTypeDoc({
          entryPoints: ["../js/src/index.ts"],
          tsconfig: "../js/tsconfig.json",
          typeDoc: {
            excludePrivate: true,
            excludeInternal: true,
            skipErrorChecking: true,
          },
          sidebar: {
            label: "JS API",
            collapsed: false,
          },
        }),
      ],
      sidebar: [
        { label: "Introduction", slug: "index" },
        {
          label: "Guide",
          items: [
            { label: "Calibration Board", slug: "guides/board" },
            { label: "Tag Generation", slug: "guides/tag" },
            { label: "Tag Detection", slug: "guides/detection" },
            { label: "MuJoCo Visualisation", slug: "guides/mujoco" },
            { label: "Recording", slug: "guides/recorder" },
          ],
        },
        {
          label: "Python API",
          link: "/python-api/index.html",
        },
        typeDocSidebarGroup,
      ],
      customCss: ["./src/styles/custom.css"],
    }),
  ],
});
